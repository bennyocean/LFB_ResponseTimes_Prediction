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
pages = ["Home", "Data", "Model", "Results", "About"]
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

# Results
if page == pages[3]:
    st.title("Results Comparison & Interpretability")
    st.markdown("""
                - Classification report (recall / prediciton)
                - Interpretability (SHAP values, marginal contribution of features)
                - limitations / contributions / future avenues
                - Extra: deep-learning model
                """)
    # Ensure the tabs directory is in the Python path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'tabs'))
    import results
    
    st.subheader('Evaluation Summary')
    
    results.comp_table()
    results.display_metrics_plot()
    results.plot_recall_comp()

    st.subheader("2. Prediction")
    st.write('Some introduction..')
    
    # Make Predictions
    y_pred, pca_feature_names, X_test_pca = results.prediction()
    
    # Load True Y Labels
    y_true = results.load_y_test_pca()
    
    # Create Prediction DataFrame
    X_test_pca_df = X_test_pca.copy()
    X_test_pca_df.columns = pca_feature_names
    X_test_pca_df['Prediction'] = y_pred

    # Make Prediction Button
    if st.button('Make Prediction'):
        st.write("Prediction:")
        st.dataframe(X_test_pca_df)


    # Prediction Plot Dropdown
    plot_type = st.selectbox(
        "Choose a plot to display:",
        ["Select Plot", "Confusion Matrix", "Prediction Distribution", "PCA Component Plot"]
        )

    if plot_type == "Confusion Matrix":
        fig = results.plot_confusion_matrix(y_true, y_pred)
        st.pyplot(fig)
    elif plot_type == "Prediction Distribution":
        fig = results.plot_pred_dist(y_pred)
        st.pyplot(fig)
    elif plot_type == "PCA Component Plot":
        pc_x = st.selectbox("Select X-axis PC", pca_feature_names, index=0)
        pc_y = st.selectbox("Select Y-axis PC", pca_feature_names, index=1)
        fig = results.plot_pca_components(X_test_pca, pc_x, pc_y)
        st.pyplot(fig)
    else:
        st.write("Please select a plot type.")


    st.subheader("3. Interpretability")
    st.write('Some introduction..')

    # SHAP Summary Plots
    shap_model_name = st.selectbox(
        "Choose a model to display SHAP values:",
        ["Select", "XGBoost", "Random Forest", "Logistic Regression"]
        )

    if shap_model_name != "Select":
        fig = results.plot_shap_values(X_test_pca, pca_feature_names, shap_model_name)
        st.pyplot(fig)


    # Display mean_abs_shap_comparison
    st.write('Some description..')
    mean_abs_shap_comparison = results.plot_shap_across_models()
    if mean_abs_shap_comparison:
        st.image(mean_abs_shap_comparison, caption="Mean Absolute SHAP Values accros each model in the voting classifier", use_column_width=True)
    else:
        st.write("Default plot image not found.")





# About
if page == pages[4]:
    st.write("### About")
    st.markdown("""
                - Team
                - Course context
                - References
                """)

    
  
