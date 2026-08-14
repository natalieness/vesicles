# vesicles

to use the ground truth annotation tool: 

### make a conda environment and activate it (if needed)
```
conda create -n vesicles -c conda-forge python=3.11 matplotlib numpy pillow tk -y 
conda activate vesicles
```

### navigate to folder you want the repo to end up in 
cd folder_of_choice/

git clone git@github.com:natalieness/vesicles.git 
cd vesicles 

### move your desired image data into the right folder 
--- by default this should be: data/fafbv783_em/ but you can change it 

### run the generate ground truth data 
python generate_ground_truth.py 

or: python generate_ground_truth.py --input-dir path_to_data/
or if you want to reannotate already annotated images: python generate_ground_truth.py --redo
