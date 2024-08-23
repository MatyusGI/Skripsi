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
python preprocesssing_project.py
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
