import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
import shap
import joblib
import gdown
import os
import sys


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
    st.write("Data")
    st.markdown("""
                - Different origins / sources
                - Pre-processing
                - Feature engineering (PCA)
                - Data visualization, statistical analysis & hypothesis tests
                """)
    

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

    sys.path.append(os.path.join(os.path.dirname(__file__), 'tabs'))
    import results
    
    results.load_eval_functions()
    results.load_pred_functions()
    results.load_interpret_functions()
    

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

    
  
