# Installation </br>
## With Anaconda </br>
dependencies in `project-env.yml`.
```bash
git clone https://github.com/MatyusGI/Skripsi
cd Skripsi
conda env create -f project-env.yml
conda activate project
```

## Run
required Rscript (change the Rscript path in generate_maps function if need to make new maps)
```bash
python Project.py
```

## Requirements
* `vision-mamba`
* `numpy`
* `pandas`
* `optuna`
* `scikit-learn`
* `torch`
* `matplotlib`
* `adabelief-pytorch`
* `biopython`

## Example Notebook
You can find example notebooks in [notebooks_project](https://github.com/MatyusGI/Skripsi/edit/main/notebooks_project) folder:
* [demo_notebook_project](https://github.com/MatyusGI/Skripsi/blob/main/notebooks_project/demo_notebook_project.ipynb)

## Notes
* To make new maps, you need to download all structures from SabDAb
* you can find the dataset used in this code in [test_data](https://github.com/MatyusGI/Skripsi/tree/main/notebooks/test_data)
* This code needs to run in the minimum specification of 8 GB VRAM GPU 
