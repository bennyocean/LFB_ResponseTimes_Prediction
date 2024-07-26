import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'tabs'))
import eda
import results
import conclusion

# Streamlit UI
st.sidebar.title("Table of Contents")
pages = ["Home", "Data", "Model", "Prediction", "Conclusion", "About"]
page = st.sidebar.radio("Go to", pages)

# Home
if page == pages[0]:
    st.title("LFB Response Time")
    st.markdown("""
                - Context / motivation / objectives
                - Data / methods used
                - Summary of results (key findings)
                """)


# Data
if page == pages[1]:
    eda.show_data_exploration()
    

# Model
if page == pages[2]:
    st.write("### Model")
    st.markdown("""
                - Regression model
                - Classification model (balancing data / recall)
                - Ensemble Methods (Voting classifier)
                - Extra: deep-learning model
                """)


# Prediction
if page == pages[3]:
    st.title("Prediction & Interpretation")
   
    #results.load_eval_functions()
    results.load_pred_functions()
    results.load_interpret_functions()
    results.load_pc_load_functions()
    
    
# Conclusion
if page == pages[4]:
    st.title("Conclusion")

    conclusion.conclusion_text()


# About
if page == pages[5]:
    st.title("About")
    
    st.write("""
            The project **Classification of London Fire Brigade Response Times** is a capstone project from the Data Scientist bootcamp at [DataScientest.com](https://datascientest.com), in cooperation with [Panthéon-Sorbonne University](https://www.pantheonsorbonne.fr/).
            """)

    st.write("### Project Members:")
    st.write("""
            - **Ismarah MAIER** [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue)](https://www.linkedin.com/in/ismarah-maier-18496613b/)
            - **Clemens PAULSEN** [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue)](https://www.linkedin.com/in/benjaminschellinger/)
            - **Dr. Benjamin SCHELLINGER** [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue)](https://www.linkedin.com/in/clemens-paulsen-a65a5a155/)
            """)

    st.write("### Project Mentor:")
    st.write("""
    - **Yazid MSAADI** (DataScientest) [![Email](https://img.shields.io/badge/Email-red)](mailto:yazid.m@datascientest.com)
    """)

    st.write("### Github:")
    st.write("[LFB project](https://github.com/DataScientest-Studio/MAY24_BDS_INT_Fire_Brigade.git)")

    st.write("### References:")
    st.write("[tbd](https://asrs.arc.nasa.gov/)")


    
  
