import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'tabs'))
from tabs import eda
import results

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
   
    results.load_eval_functions()
    results.load_pred_functions()
    results.load_interpret_functions()
    results.load_pc_load_functions()
    

# Conclusion
if page == pages[4]:
    st.title("Conclusion")


# About
if page == pages[5]:
    st.write("### About")
    st.markdown("""
                - Team
                - Course context
                - References
                """)

    
  
