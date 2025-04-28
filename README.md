# Generative T2EG
Software designed to convert MRI diffusion-weighted sequences into T2EG sequences

### Prerequisites
Python 3.8

### Usage
First, clone the git directory :

```
git clone https://github.com/NeuroSainteAnne/GenerativeT2EG.git
cd GenerativeT2EG/
```

Then install required modules:

```
pip install -r requirements.txt
pip install jupyterlab
```

Open the jupyter notebook

```
jupyter notebook
```

### Data preparation

Open the [genT2EG-Dataprep](genT2EG-Dataprep.ipynb) in order to convert you NIFTI files in a Memmap format for model training

### Model training

Open the [genT2EG-Train](genT2EG-Train.ipynb) notebook and follow instructions to train and save your model.

Alternatively, download the pre-trained weights available [here](https://github.com/NeuroSainteAnne/GenerativeT2EG/releases/tag/1.0).

### Inference

Once the model is trained and saved, you can convert 4D NIFTI DWI volumes into genT2EG volumes with the [genT2EG-Convert](genT2EG-Convert.ipynb) tool.

