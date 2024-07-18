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


# Streamlit UI
st.title("LFB Response Time")
st.sidebar.title("Table of Contents")
pages = ["Home", "Data", "Model", "Results", "About"]
page = st.sidebar.radio("Go to", pages)

# Home
if page == pages[0]:
    st.write("### Home")
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

# Results
if page == pages[3]:
    st.write("### Results")
    st.markdown("""
                - Classification report (recall / prediciton)
                - Interpretability (SHAP values, marginal contribution of features)
                - limitations / contributions / future avenues
                - Extra: deep-learning model
                """)

# About
if page == pages[4]:
    st.write("### About")
    st.markdown("""
                - Team
                - Course context
                - References
                """)

    
  
