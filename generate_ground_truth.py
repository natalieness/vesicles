"""Manual ground-truth annotation of circular structures (e.g. vesicles) in EM images.

Usage
-----
    python generate_ground_truth.py
    python generate_ground_truth.py --input-dir data/fafb_em --output-dir data/ground_truth
    python generate_ground_truth.py --redo          # re-annotate images that already have a csv

Controls
--------
    left click + drag    draw a circle (press at centre, drag out to the radius)
    right click          delete the circle under the cursor
    delete / backspace   delete the circle under the cursor
    u                    undo the last circle
    scroll wheel         zoom in / out around the cursor
    middle click + drag  pan
    r                    reset the view
    (the matplotlib toolbar pan/zoom tools also work; drawing is disabled while they are active)

Buttons
-------
    Save & continue      write csv + overlay, then open the next un-annotated image
    Save & stop          write csv + overlay, then close the viewer
    Cancel               close without saving the current image
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

# The macosx backend segfaults when a callback touches figures while its event
# loop is running, so use Tk unless the user has explicitly asked for something
# else via MPLBACKEND.
if not os.environ.get("MPLBACKEND"):
    for _backend in ("TkAgg", "QtAgg"):
        try:
            matplotlib.use(_backend)
            break
        except Exception:  # pragma: no cover - depends on local install
            continue

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.widgets import Button
from PIL import Image, ImageDraw

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

MIN_RADIUS = 1.0  # px, anything smaller is treated as a stray click
CIRCLE_COLOR = "#00ff88"
HIGHLIGHT_COLOR = "#ff2d55"
PREVIEW_COLOR = "#ffd000"
CSV_HEADER = ["index", "image", "x", "y", "radius"]


def list_images(input_dir):
    """All image files in `input_dir`, sorted by name."""
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def csv_path(output_dir, image_path):
    return output_dir / f"{image_path.stem}_anns.csv"


def overlay_path(output_dir, image_path):
    return output_dir / f"{image_path.stem}_overlay.png"


def pending_images(input_dir, output_dir, redo=False):
    """Images that still need annotating (i.e. have no csv yet), in order."""
    images = list_images(input_dir)
    if redo:
        return images
    return [p for p in images if not csv_path(output_dir, p).exists()]


def load_image(path):
    """Return (array for display, RGB PIL image for the overlay)."""
    pil_image = Image.open(path)
    pil_image.load()
    if pil_image.mode not in {"L", "RGB"}:
        pil_image = pil_image.convert("L" if pil_image.mode in {"1", "I;16", "I"} else "RGB")
    return np.asarray(pil_image), pil_image.convert("RGB")


def write_csv(path, image_path, circles):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for i, (x, y, r) in enumerate(circles):
            writer.writerow([i, image_path.name, f"{x:.3f}", f"{y:.3f}", f"{r:.3f}"])


def write_overlay(path, pil_image, circles, supersample=4, linewidth=1.0):
    """Draw the circles onto a copy of the source image and save it.

    Done with PIL rather than a matplotlib figure: creating and destroying a
    figure from inside a GUI button callback segfaults the macosx backend, and
    this keeps the overlay exactly the same pixel size as the source image.
    The outlines are drawn on a transparent layer at `supersample` resolution
    and shrunk back down, which antialiases them without touching the EM pixels.
    """
    base = pil_image.convert("RGBA")
    s = max(1, int(supersample))
    layer = Image.new("RGBA", (base.width * s, base.height * s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for x, y, r in circles:
        # imshow puts pixel centres on integers; PIL puts them at +0.5.
        cx, cy = (x + 0.5) * s, (y + 0.5) * s
        rs = r * s
        draw.ellipse(
            [cx - rs, cy - rs, cx + rs, cy + rs],
            outline=CIRCLE_COLOR,
            width=max(1, round(linewidth * s)),
        )
    layer = layer.resize(base.size, resample=Image.LANCZOS)
    Image.alpha_composite(base, layer).convert("RGB").save(path)


class CircleAnnotator:
    """Interactive viewer that walks through a list of images one at a time."""

    def __init__(self, images, output_dir):
        self.images = list(images)
        self.output_dir = output_dir
        self.position = 0

        self.image = None
        self.pil_image = None
        self.circles = []       # list of [x, y, r]
        self.patches = []       # matching list of Circle artists
        self.preview = None
        self.press_xy = None
        self.cursor_xy = None
        self.highlighted = None
        self.pan_start = None

        self.fig = plt.figure(figsize=(9, 9.6))
        self.ax = self.fig.add_axes([0.06, 0.14, 0.88, 0.80])
        self.ax.set_facecolor("black")

        self.status = self.fig.text(
            0.5, 0.065, "", ha="center", va="center", fontsize=9, color="#444444"
        )

        self.buttons = []
        for label, left, color, callback in [
            ("Save && continue", 0.10, "#cdebd6", self.on_save_continue),
            ("Save && stop", 0.39, "#cddceb", self.on_save_stop),
            ("Cancel", 0.68, "#ebcdcd", self.on_cancel),
        ]:
            axes = self.fig.add_axes([left, 0.015, 0.22, 0.045])
            button = Button(axes, label, color=color, hovercolor="#ffffff")
            button.on_clicked(callback)
            self.buttons.append(button)

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.load_current()

    # ------------------------------------------------------------------ state

    @property
    def current_path(self):
        return self.images[self.position]

    def load_current(self):
        path = self.current_path
        self.image, self.pil_image = load_image(path)
        self.circles = []
        self.patches = []
        self.preview = None
        self.highlighted = None

        self.ax.clear()
        self.ax.imshow(self.image, cmap="gray", interpolation="nearest")
        self.ax.set_title(
            f"[{self.position + 1}/{len(self.images)}]  {path.name}",
            fontsize=11,
        )
        self.ax.set_xlabel("drag = new circle | right click / delete = remove | u = undo | scroll = zoom")
        self.ax.xaxis.label.set_fontsize(8)
        self.ax.xaxis.label.set_color("#666666")
        self.reset_view()
        self.update_status()

    def reset_view(self):
        height, width = self.image.shape[:2]
        self.ax.set_xlim(-0.5, width - 0.5)
        self.ax.set_ylim(height - 0.5, -0.5)
        self.ax.set_aspect("equal")
        self.fig.canvas.draw_idle()

    def update_status(self, message=None):
        text = f"{len(self.circles)} annotation(s) on this image"
        remaining = len(self.images) - self.position - 1
        if remaining > 0:
            text += f"   |   {remaining} image(s) left after this one"
        if message:
            text += f"   |   {message}"
        self.status.set_text(text)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------- annotation

    def add_circle(self, x, y, r):
        patch = Circle((x, y), r, fill=False, edgecolor=CIRCLE_COLOR, linewidth=1.4)
        self.ax.add_patch(patch)
        self.circles.append([x, y, r])
        self.patches.append(patch)

    def remove_circle(self, index):
        self.patches[index].remove()
        del self.patches[index]
        del self.circles[index]
        if self.highlighted == index:
            self.highlighted = None
        elif self.highlighted is not None and self.highlighted > index:
            self.highlighted -= 1

    def circle_at(self, x, y, tolerance=6.0):
        """Index of the circle under (x, y): the smallest one containing the point,
        otherwise the one whose outline is closest (within `tolerance` px)."""
        inside = []
        near_edge = []
        for i, (cx, cy, r) in enumerate(self.circles):
            distance = float(np.hypot(x - cx, y - cy))
            if distance <= r:
                inside.append((r, i))
            elif distance - r <= tolerance:
                near_edge.append((distance - r, i))
        if inside:
            return min(inside)[1]
        if near_edge:
            return min(near_edge)[1]
        return None

    def highlight(self, index):
        if index == self.highlighted:
            return
        if self.highlighted is not None and self.highlighted < len(self.patches):
            self.patches[self.highlighted].set_edgecolor(CIRCLE_COLOR)
            self.patches[self.highlighted].set_linewidth(1.4)
        if index is not None:
            self.patches[index].set_edgecolor(HIGHLIGHT_COLOR)
            self.patches[index].set_linewidth(2.0)
        self.highlighted = index
        self.fig.canvas.draw_idle()

    def delete_under_cursor(self):
        if self.cursor_xy is None:
            return
        index = self.circle_at(*self.cursor_xy)
        if index is None:
            self.update_status("nothing to delete under the cursor")
            return
        self.remove_circle(index)
        self.update_status("deleted 1 annotation")

    # ----------------------------------------------------------------- events

    def toolbar_active(self):
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        return bool(getattr(toolbar, "mode", ""))

    def on_press(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if event.button == 2:
            self.pan_start = (event.xdata, event.ydata)
            return
        if self.toolbar_active():
            return
        if event.button == 1:
            self.press_xy = (event.xdata, event.ydata)
            self.preview = Circle(
                self.press_xy, 0.0, fill=False,
                edgecolor=PREVIEW_COLOR, linewidth=1.4, linestyle="--",
            )
            self.ax.add_patch(self.preview)
        elif event.button == 3:
            index = self.circle_at(event.xdata, event.ydata)
            if index is None:
                self.update_status("nothing to delete there")
            else:
                self.remove_circle(index)
                self.update_status("deleted 1 annotation")

    def on_motion(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        self.cursor_xy = (event.xdata, event.ydata)

        if self.pan_start is not None:
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            self.ax.set_xlim(x0 - dx, x1 - dx)
            self.ax.set_ylim(y0 - dy, y1 - dy)
            self.fig.canvas.draw_idle()
            return

        if self.preview is not None and self.press_xy is not None:
            radius = float(np.hypot(event.xdata - self.press_xy[0],
                                    event.ydata - self.press_xy[1]))
            self.preview.set_radius(radius)
            self.fig.canvas.draw_idle()
            return

        self.highlight(self.circle_at(event.xdata, event.ydata))

    def on_release(self, event):
        if event.button == 2:
            self.pan_start = None
            return
        if event.button != 1 or self.preview is None:
            return

        radius = self.preview.get_radius()
        centre = self.press_xy
        self.preview.remove()
        self.preview = None
        self.press_xy = None

        if radius >= MIN_RADIUS:
            self.add_circle(centre[0], centre[1], radius)
            self.update_status()
        else:
            self.update_status("circle too small - drag further out from the centre")

    def on_scroll(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(
            event.xdata + (x0 - event.xdata) * scale,
            event.xdata + (x1 - event.xdata) * scale,
        )
        self.ax.set_ylim(
            event.ydata + (y0 - event.ydata) * scale,
            event.ydata + (y1 - event.ydata) * scale,
        )
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in {"delete", "backspace"}:
            self.delete_under_cursor()
        elif event.key in {"u", "ctrl+z"}:
            if self.circles:
                self.remove_circle(len(self.circles) - 1)
                self.update_status("undid last annotation")
            else:
                self.update_status("nothing to undo")
        elif event.key == "r":
            self.reset_view()

    # ---------------------------------------------------------------- buttons

    def save_current(self):
        path = self.current_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        circles = [tuple(c) for c in self.circles]
        write_csv(csv_path(self.output_dir, path), path, circles)
        write_overlay(overlay_path(self.output_dir, path), self.pil_image, circles)
        print(
            f"saved {len(circles)} annotation(s) for {path.name} -> "
            f"{csv_path(self.output_dir, path).name}, "
            f"{overlay_path(self.output_dir, path).name}"
        )

    def on_save_continue(self, _event):
        self.save_current()
        if self.position + 1 >= len(self.images):
            print("no images left to annotate")
            plt.close(self.fig)
            return
        self.position += 1
        self.load_current()

    def on_save_stop(self, _event):
        self.save_current()
        plt.close(self.fig)

    def on_cancel(self, _event):
        print(f"cancelled - {self.current_path.name} was not saved")
        plt.close(self.fig)

    def run(self):
        # block=True matters under IPython, where interactive mode would
        # otherwise let show() return immediately and drop the window.
        plt.show(block=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("data/fafb_em"),
                        help="folder of images to annotate (default: data/fafb_em)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/fafb_em_gt"),
                        help="folder for {name}_anns.csv and {name}_overlay.png "
                             "(default: data/fafb_em_gt)")
    parser.add_argument("--redo", action="store_true",
                        help="also show images that already have annotations")
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        parser.error(f"input directory not found: {args.input_dir}")

    images = pending_images(args.input_dir, args.output_dir, redo=args.redo)
    if not images:
        total = len(list_images(args.input_dir))
        if total == 0:
            print(f"no images found in {args.input_dir}")
        else:
            print(f"all {total} image(s) in {args.input_dir} are already annotated "
                  f"in {args.output_dir} (use --redo to annotate them again)")
        return 0

    print(f"{len(images)} image(s) to annotate; saving to {args.output_dir}")
    CircleAnnotator(images, args.output_dir).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
