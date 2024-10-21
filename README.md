## Introduction

This repository contains the code for our project : **Response Time Prediction for the LONDON FIRE BRIGADE**. It has been developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/), in cooperation with [Panthéon-Sorbonne University](https://www.pantheonsorbonne.fr/).

We aim to develop a machine learning model to predict the response times of the [London Fire Brigade (LFB)](https://www.london-fire.gov.uk/), thus, improving the operational efficiency and contributing to the economic and scientific advancement of emergency response services.

This project was developed by the following team :

- Ismarah MAIER ([GitHub](https://github.com/isi-pizzy) / [LinkedIn](https://www.linkedin.com/in/ismarah-maier-18496613b/))
- Clemens PAULSEN ([GitHub](https://github.com/ClemensPaulsen) / [LinkedIn](https://www.linkedin.com/in/clemens-paulsen-a65a5a155/))
- Dr. Benjamin SCHELLINGER ([GitHub](https://github.com/bennyocean) / [LinkedIn](https://www.linkedin.com/in/benjaminschellinger/))

You can browse and run the [notebooks](./notebooks). 

## Streamlit App

### 1. Create a Virtual Environment
Navigate to the root of the project directory, and create a virtual environment using Python 3.10:

To use the app locally, follow the steps below to set up the environment and install the necessary dependencies.

```shell
python3.10 -m venv venv
```

### 2. Activate the Virtual Environment

Activate the virtual environment:
- On macOS/Linux:
    ```shell
    source venv/bin/activate
    ```
- On Windows:
    ```shell
    venv\Scripts\activate
    ```

### 3. Install Dependencies
Once the virtual environment is activated, install the dependencies from the requirements.txt file:

```shell
pip install -r requirements.txt
```
This will ensure that all the necessary packages are installed with the correct versions.

### 4. Run Streamlit App
To run the app, make sure your virtual environment is activated:

```shell
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```
Then run the Streamlit app:

```shell
streamlit run streamlit_app/lfb_streamlit.py
```
That's it! The app should now be running locally at [localhost:8501](http://localhost:8501) on your machine.

### 5. (Optional) Live App
If you want to see the app without going through the installation process, simply visit the live Streamlit app hosted on Heroku:
[LFB Response Times App](https://lfb-response-times-1319b829624f.herokuapp.com/)

#### Quick Walkthrough:
- **Home:**
A brief introduction to the application.
- **Data:**
Insights into data analysis, statistics, feature engineering, and target variable creation.
- **Model:**
Presentation of models selected and hyperparameters tuning.
- **Prediction:**
Here you can predict response times and examine the model’s interpretability.
- **Conclusion:**
Summary and key take aways.
- **About:**
Background information on the project.

#### Have fun! :)

## MLOps

Please refere to the [MLOps](https://github.com/bennyocean/LFB_MLOps.git) repo for getting insights to the deployment, automation, and monitoring of our ML models.