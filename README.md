# Goal
Design and implement  an auto-complete server using scripts from all the episodes of Seinfeld. The server will try to predict how Jerry Seinfeld would finish a given incomplete sentence.

#### Dataset obtained from https://www.kaggle.com/thec03u5/seinfeld-chronicles

## To run the code on a computer (with CPU)

### Git clone the code
```
git clone https://github.com/ArbinTimilsina/AutoCompleteJerry.git
cd AutoComplete
```

### Create a conda environment (Python 3)
```
conda update -n base conda
conda env create -f requirements/cpu_requirements.yml
conda activate envAutoComplete
```

### Switch Keras backend to TensorFlow
```
KERAS_BACKEND=tensorflow python -c "from keras import backend"
```

### Create an IPython kernel for the environment
```
python -m ipykernel install --user --name envAutoCompleteJerry --display-name "envAutoCompleteJerry"
```

### To train the model, do
```
python train_model.py
``` 

### To use the model, do
```
python run_server.py
```
and then curl the following in a new terminal (or paste into a web browser)
```
http://localhost:5050/autocomplete?seed=What+is+deal
```

You will get output similar to 

<img src="plots/output.png" style="width: 500px;"/>

Replace the string after ```?seed=``` to change the seed and see suggested completions!

###  To experiment and play with a copy of the inner workings, do
```
jupyter notebook model_creation_playground.ipynb
```
Make sure to change the kernel to envAutoComplete using the drop-down menu (Kernel > Change kernel > envAutoCompleteJerry)
