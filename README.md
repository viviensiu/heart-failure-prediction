## Problem Description

## Dataset
* [Kaggle Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/)

## EDA
* Refer Jupyter Notebook [part_1_preprocessing.ipynb](https://github.com/viviensiu/heart-failure-prediction/blob/main/notebook/part_1_preprocessing.ipynb).

## Model training
* Refer Jupyter Notebook [part_2_modeling.ipynb](https://github.com/viviensiu/heart-failure-prediction/blob/main/notebook/part_2_modeling.ipynb).

## Exporting notebook to script
* Refer script [training.py](https://github.com/viviensiu/heart-failure-prediction/blob/main/script/training.py).

## Model deployment
* 

## Dependency and environment management
### [Optional] Create New Conda Virtual Environment 
* `conda create -n ml-midterm-env` then `conda activate ml-midterm-env`
* `conda install pip`

### Create Virtual Environment using Pipenv
* Make sure you have pipenv in your existing environment, if not, first execute `pip install pipenv`.
* If you already installed pipenv, execute `pipenv install numpy scikit-learn seaborn jupyter notebook xgboost streamlit`.
* If it's successful, you should see both `Pipfile` and `Pipfile.lock`.

### Activate virtual env
* To use the pipenv environment created from previous section , navigate to the project folder which contains `Pipfile` and `Pipfile.lock`, then execute `pipenv shell`.
* Execute the following commands after `pipenv shell` for:
    * Open Jupyter Notebook: `jupyter notebook`

## Containerization

## Cloud Deployment

## Evaluation Criteria
The project will be evaluated using these criteria:
* Problem description
* EDA
* Model training
* Exporting notebook to script
* Model deployment
* Reproducibility
* Dependency and environment management
* Containerization
* Cloud deployment

See ["Criteria and points award system"](https://docs.google.com/spreadsheets/d/e/2PACX-1vQCwqAtkjl07MTW-SxWUK9GUvMQ3Pv_fF8UadcuIYLgHa0PlNu9BRWtfLgivI8xSCncQs82HDwGXSm3/pubhtml).