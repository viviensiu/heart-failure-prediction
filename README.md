## Problem Description
<p align="center">
    <!--img src="https://github.com/viviensiu/LLM-project/blob/main/image/problem.jpg" width=200 -->
   <img src="https://github.com/viviensiu/heart-failure-prediction/blob/main/img/kenny-eliason-MEbT27ZrtdE-unsplash.jpg">
</p>
<p align="center">
   <em>Image credits: <a href="https://unsplash.com/@neonbrand">Kenny Eliason on Unsplash</a></em>
</p>
> Cardiovascular diseases (CVD) are one of the leading cause of deaths worldwide. By examining the common risk factors, such as high blood cholestrol, chest pains, age risks and other factors featured in this dataset, a person's risk of having CVD could be detected earlier, thus reducing the number of deaths caused by CVD.

> Using a Machine Learning approach, this project's goal is to help researchers in identifying the importance and correlations of each of the risk factors mentioned above using existing CVD medical records. A Machine Learning model trained on this dataset could be used to predict if a new patient is potentially at risk of having heart disease.

## Dataset
* [Kaggle Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/)

## EDA
* Refer Jupyter Notebook [part_1_preprocessing.ipynb](https://github.com/viviensiu/heart-failure-prediction/blob/main/notebook/part_1_preprocessing.ipynb).

## Model training
* Refer Jupyter Notebook [part_2_modeling.ipynb](https://github.com/viviensiu/heart-failure-prediction/blob/main/notebook/part_2_modeling.ipynb).

## Exporting notebook to script
* Refer script [training.py](https://github.com/viviensiu/heart-failure-prediction/blob/main/training.py).

## Model deployment
* Model is deployed with two separate options: Flask and Streamlit.
    * Flask:
    * Streamlit: runs on local workstation and on Streamlit Cloud. Link is provided at [Cloud Deployment]().
* Note that either one can be used for own convenience.

## Dependency and environment management
The following steps are for reproducing the results of this repo on your local workstation.

### Reproducing this repo
* Create a new folder called `heart_failure_prediction` on your local workstation.
* In command prompt/terminal, navigate to this new folder. Then clone this repo to this new folder using `git clone https://github.com/viviensiu/heart-failure-prediction.git`. 
* Alternatively you could use these options to clone this repo.
    * VSCode or 
    * Github Desktop

### [Optional] Install pip
* Usually pip is included if you have installed python. To check this, execute `pip --version` in command prompt/terminal. If you could see a version, pip is installed.
* If not, refer this installation guide: [Official pip installation guidelines](https://pip.pypa.io/en/stable/installation/).

### [Optional] Install pipenv
* `pip install pipenv`

### Create Virtual Environment using Pipenv
* If you cloned this repo, `Pipfile` and `Pipfile.lock` should already be available in `heart_failure_prediction`. You can skip the following step.
* If not, navigate to `heart_failure_prediction` in your command prompt/terminal: 
    * Execute `pipenv install numpy scikit-learn seaborn jupyter notebook xgboost streamlit flask requests`. 
    * If it's successful, you should see both `Pipfile` and `Pipfile.lock`.

### Activate virtual env
* To activate the pipenv environment defined in `Pipfile` and `Pipfile.lock`, in command prompt/terminal: 
    * Navigate to `heart_failure_prediction`. 
    * `pipenv shell`.
* Execute the following commands after `pipenv shell` to:
    * Open Jupyter Notebook: `jupyter notebook`.
    * Run `training.py` script: `python training.py`.
    * Run heart disease prediction app: `streamlit run heart_disease_prediction.py`.

## Containerization

## Cloud Deployment
* This prediction app is deployed to Streamlit Cloud, here's the url to try it: [https://heart-vs.streamlit.app/](https://heart-vs.streamlit.app/).
* The following steps are for those who are interested to replicate deployment to Streamlit Cloud:
    * Sign in to [Streamlit Cloud](https://streamlit.io/cloud) with Github account.
    * Follow the [instructions here](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) to create a Streamlit app by specifying the Github repo, Streamlit app (for this repo, use [`heart_disease_prediction.py`](https://github.com/viviensiu/heart-failure-prediction/blob/main/heart_disease_prediction.py))  and provide a file for environment setup ([Pipfile](https://github.com/viviensiu/heart-failure-prediction/blob/main/Pipfile) was used here).
    * 

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