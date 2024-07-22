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
    # Introduction to the prediction part
    st.write("""
    ### Prediction
    Our binary classification model aims to predict the response time mode based on various principal components derived from the input features. Click the "Make Prediction" button below to generate predictions for the test dataset, which consists of 307,541 data points retained from training to evaluate model generalization on unseen data.
    """)

    if st.button('Make Prediction'):
        st.write("Prediction:")
        st.dataframe(X_test_pca_df)

        # Interpretation of the Prediction DataFrame
        st.write("""
        The table above shows the principal components used for the prediction and the corresponding predicted response time mode (0 or 1). The 'Prediction' column indicates whether the response time is classified as not reached the target of 6 minutes (i.e., class 0) or reached the 6 minute goal (i.e., class 1) by the model. These predictions are made on the test set to assess how well the model generalizes to new, unseen data.
        """)

    # Prediction Plot Dropdown
    st.markdown("""
    #### Prediction Evaluation:
    - **Confusion Matrix**: Visualizes the performance of the classification model by showing the number of correct and incorrect predictions for each class, helping to identify how well the model distinguishes between different classes.
    - **Distribution of Predictions**: Displays the frequency of predicted classes, providing insight into the overall prediction distribution and helping to identify any class imbalance in the model's predictions.
    - **PCA Component Plot**: The PCA component plot provides a visual indication whether the model is effectively using the principal components to distinguish between the two classes.
                """)
    
    plot_type = st.selectbox(
        "Choose a plot to display:",
        ["Select Plot", "Confusion Matrix", "Prediction Distribution", "PCA Component Plot"]
        )

    if plot_type == "Confusion Matrix":
        fig = results.plot_confusion_matrix(y_true, y_pred)
        st.pyplot(fig)
        st.write("""
        The confusion matrix shows that the model correctly predicted 155,420 instances of 'Reached (<=6 min)' but misclassified 61,878 instances of 'Reached (<=6 min)' as 'Not Reached (>6 min)', representing a considerable number of false negatives.
        """)
    elif plot_type == "Prediction Distribution":
        fig = results.plot_pred_dist(y_pred)
        st.pyplot(fig)
        st.write("""
        The distribution plot indicates that the model predicts more instances of 'Goal Reached (<=6 min)' compared to 'Goal not Reached (>6 min)', providing insight into the model's tendency and potential class imbalance in the predictions.
        """)
    elif plot_type == "PCA Component Plot":
        pc_x = st.selectbox("Select X-axis PC", pca_feature_names, index=0)
        pc_y = st.selectbox("Select Y-axis PC", pca_feature_names, index=1)
        fig = results.plot_pca_components(X_test_pca_df, pc_x, pc_y)
        st.pyplot(fig)
    else:
        st.write("Please select a plot type.")


    st.subheader("3. Model Interpretation")
    # Introduction to Interpretability
    st.write("""
    Model interpretability is crucial for understanding how our classification model makes predictions. It helps us identify which features are most influential in determining the response time mode, ensuring transparency and trust in the model's decisions. By analyzing interpretability metrics such as SHAP values, we can gain insights into the model's behavior, detect potential biases, and improve model performance by focusing on key features.
    """)

    st.write('#### Mean Absolute SHAP Values')
    st.write("""
    Mean Absolute SHAP Values accros each model in the voting classifier
             """)
    
    # Display mean_abs_shap_comparison
    mean_abs_shap_comparison = results.plot_shap_across_models()
    if mean_abs_shap_comparison:
        st.image(mean_abs_shap_comparison, caption="Mean absolute SHAP values accros each model in the voting classifier", use_column_width=True)
    else:
        st.write("Default plot image not found.")
    st.write('Some interpretation..')

    # SHAP Summary Plots
    st.write("""
    The SHAP summary plot explains the contribution of each principal component to the model's predictions. Higher SHAP values indicate greater importance in predicting whether the response time goal is reached. This helps us understand which features most influence the model's decision-making process in our classification problem.
            """)
    shap_model_name = st.selectbox(
        "Choose a model to display SHAP values:",
        ["Select", "XGBoost", "Random Forest", "Logistic Regression"]
    )

    if shap_model_name != "Select":
        shap_plot = results.plot_shap_values(shap_model_name)
        if shap_plot:
            st.image(shap_plot, caption=f"SHAP summary plot for {shap_model_name}", use_column_width=True)
        else:
            st.write("SHAP summary plot not found.")
    
    # SHAP Violin Plots
    st.write('#### Violin Plots of SHAP Values')
    st.write("""
    The SHAP violin plot explains the distribution and impact of the principal components on the model's predictions. Each violin represents the distribution of SHAP values for a feature, indicating both the magnitude and the direction (positive or negative) of the feature's influence on the prediction. This visualization helps in understanding which features are most important in determining the response time mode.
             """)
    
    # Display violin plot of voting classifier
    voilin_votingclf = results.plot_violin_votingclf()
    if voilin_votingclf:
        st.image(voilin_votingclf, caption="Violin plot of SHAP values for combined models (Voting Classifier)", use_column_width=True)
    else:
        st.write("Default plot image not found.")
    st.write('Some interpretation..')

    shap_model_name = st.selectbox(
        "Choose a model to display SHAP violin plot:",
        ["Select", "XGBoost", "Random Forest", "Logistic Regression"]
    )

    if shap_model_name != "Select":
        shap_plot = results.plot_shap_violins(shap_model_name)
        if shap_plot:
            st.image(shap_plot, caption=f"SHAP violin plot for {shap_model_name}", use_column_width=True)
        else:
            st.write("SHAP violin plot not found.")
    
    # SHAP Local Interpretation
    st.write('#### Local Interpretation of Model Prediction')
    st.write("""
    The local interpration....
            """)
    shap_model_name = st.selectbox(
        "Choose a model for local interpretation:",
        ["Select", "XGBoost", "Random Forest", "Logistic Regression", "Combined Voting Classifier"]
    )
    
    observation_number = st.number_input(
        f"Enter the observation number (between 0 and {X_test_pca.shape[0]-1}):",
        min_value=0, 
        max_value=X_test_pca.shape[0]-1,
        step=1
    )

    # Display the SHAP force plot for the selected model and observation
    if shap_model_name != "Select":
        if st.button('Generate SHAP Force Plot'):
            force_plot_path = results.local_interpretation(shap_model_name, X_test_pca, pca_feature_names, index=observation_number)
            if force_plot_path:
                st.image(force_plot_path, caption=f"SHAP Force Plot for {shap_model_name} (Observation {observation_number})", use_column_width=True)
            else:
                st.write("SHAP force plot not found or error in generating the plot.")

    
    # Display the interactive SHAP force plot for the selected model and number of observations
    st.write('#### Interactive SHAP Plot')
    st.write("""
    In addition to only plotting the SHAP values for one observation, we can also display the marginal contribution of features to our model's prediction over a pre-defined range of observations (e.g., 1000 data points). ....
            """)
    
    # Input for the number of observations
    n_observations = st.number_input(
        "Enter the number of observations (e.g., 1000):",
        min_value=1, 
        max_value=X_test_pca.shape[0]-1,
        value=1000, 
        step=1
    )

    if shap_model_name != "Select":
        if st.button('Generate Interactive SHAP Force Plot'):
            force_plot = results.interactive_force_plot(shap_model_name, X_test_pca, pca_feature_names, n_observations=n_observations)
            if force_plot:
                st_shap(force_plot)
            else:
                st.write("SHAP force plot not found or error in generating the plot.")




# About
if page == pages[4]:
    st.write("### About")
    st.markdown("""
                - Team
                - Course context
                - References
                """)

    
  
