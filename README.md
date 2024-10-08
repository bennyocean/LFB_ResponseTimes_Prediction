## Introduction

This repository contains the code for our project : **Response Time Prediction for the LONDON FIRE BRIGADE**. It has been developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/), in cooperation with [Panthéon-Sorbonne University](https://www.pantheonsorbonne.fr/).

We aim to develop a machine learning model to predict the response times of the [London Fire Brigade (LFB)](https://www.london-fire.gov.uk/), thus, improving the operational efficiency and contributing to the economic and scientific advancement of emergency response services.

This project was developed by the following team :

- Ismarah MAIER ([GitHub](https://github.com/isi-pizzy) / [LinkedIn](https://www.linkedin.com/in/ismarah-maier-18496613b/))
- Clemens PAULSEN ([GitHub](https://github.com/ClemensPaulsen) / [LinkedIn](https://www.linkedin.com/in/clemens-paulsen-a65a5a155/))
- Dr. Benjamin SCHELLINGER ([GitHub](https://github.com/bennyocean) / [LinkedIn](https://www.linkedin.com/in/benjaminschellinger/))

## Instructions

You can browse and run the [notebooks](./notebooks). 

You will need to install the dependencies (in a dedicated environment) :

```
pip install -r requirements.txt
```

## Streamlit App

<img src="./streamlit.png" alt="Streamlit App Preview" width="800"/>

To run the app:

```shell
conda create -f environment.yml
conda activate lfb_env
# optional: pip install -r requirements.txt
streamlit run streamlit_app/lfb_streamlit.py
```

The app should then be available at [localhost:8501](http://localhost:8501).

## MLOps

In progress and soon available :)