import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from PIL import Image
import shap
import joblib
import os

# Results

### 1. Evaluation Comparison
### 1.1 Evaluation Metric Scores
def comp_table():
    data = {
        'Model': [
            'Random Forest', 
            'XGBoost', 
            'Voting Classifier (Soft)', 
            'Voting Classifier (Hard)',
            'Stacking Classifier'
        ],
        'Precision': [0.72, 0.73, 0.72, 0.72, 0.73],
        'Recall': [0.70, 0.70, 0.69, 0.69, 0.70],
        'F1-Score': [0.71, 0.71, 0.70, 0.70, 0.71],
        'ROC AUC': [0.7352, 0.7411, 0.7364, 0, 0.7403],  # 0 Placeholder for Voting Classifier (Hard)
        'Accuracy': [0.70, 0.6986, 0.6884, 0.6919, 0.6968],
        'Balanced Accuracy': [0.6733, 0.6756, 0.6756, 0.6755, 0.6768]
    }

    comp = pd.DataFrame(data)
    comp.set_index('Model', inplace=True)
    st.write("Evaluation Comparison of Final Model Scores")
    st.write('Some description..')
    st.dataframe(comp)


def display_metrics_plot():
    st.write('Some description..')
    
    # Define the relative path where the images are stored
    image_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_evaluation_comparison')

    # Dictionary to map score metrics to their respective image filenames
    score_metrics = {
        "Precision": "precision.png",
        "Recall": "recall.png",
        "F1-Score": "f1_score.png",
        "ROC AUC Score": "roc_auc.png",
        "Accuracy": "accuracy.png",
        "Balanced Accuracy": "balanced_accuracy.png"
    }

    # Create a dropdown menu for selecting the score metric with a default option
    options = ["Select a Score Metric"] + list(score_metrics.keys())
    selected_metric = st.selectbox("Select a Score Metric:", options)

    # Load and display the corresponding image if a valid option is selected
    if selected_metric != "Select a Score Metric":
        image_file = os.path.join(image_path, score_metrics[selected_metric])
        image = Image.open(image_file)
        st.image(image, caption=selected_metric, use_column_width=True)

### 1.2 Recall Evaluation (focus on minority class)

def plot_recall_comp():
    st.write('Some explanation. Focus on Recall of minority class, i.e., target of 6 minutes not reached. Total test size: 307541. Majority class ~ 217298, minority class 90243. 2Goal to reduce false negatives and thus increase recall of minority class.')

    recall_comp = {
        'Model': [
            'Random Forest', 
            'XGBoost', 
            'Voting Classifier (Soft)', 
            'Voting Classifier (Hard)',
            'Stacking Classifier'
        ],
        'Recall (Not Reached (>6 min))': [0.61, 0.62, 0.64, 0.64, 0.63],
        'Recall (Reached (<=6 min))': [0.73, 0.73, 0.71, 0.72, 0.73]
    }

    recall_comp = pd.DataFrame(recall_comp)

    # Find the model with the highest Recall (Not Reached (>6 min))
    best_model = 'Voting Classifier (Hard)'

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = range(len(recall_comp['Model']))

    # Plot Recall (Not Reached (>6 min))
    bars1 = ax.bar(index, recall_comp['Recall (Not Reached (>6 min))'], bar_width, label='Recall (Not Reached (>6 min))', color='lightcoral')

    # Plot Recall (Reached (<=6 min))
    bars2 = ax.bar([i + bar_width for i in index], recall_comp['Recall (Reached (<=6 min))'], bar_width, label='Recall (Reached (<=6 min))', color='lightgreen')

    # Highlight the bar with the highest Recall (Not Reached (>6 min))
    for i, bar in enumerate(bars1):
        if recall_comp['Model'][i] == best_model:
            bar.set_color('blue')

    # Add labels and title
    ax.set_xlabel('')
    ax.set_ylabel('Recall')
    ax.set_title('Recall Class Comparison')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(recall_comp['Model'], rotation=45, ha='right')
    ax.legend()

    # Adjust y-axis limits
    ax.set_ylim(0.55, 0.8)
    
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_evaluation_comparison', 'recall_comparison.png')
    plt.savefig(save_path)
    st.pyplot(fig)

    st.write('Some Conclusion..')


### 2 Prediction
### 2.1 Predicting Labels

def load_X_test_pca():
    parquet_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'X_test_pca.parquet')
    X_test_pca = pd.read_parquet(parquet_path)
    num_components = X_test_pca.shape[1]
    pca_feature_names = [f'PC{i+1}' for i in range(num_components)]
    return X_test_pca, pca_feature_names

def load_y_test_pca():
    parquet_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'y_test_pca.parquet')
    y_test_pca = pd.read_parquet(parquet_path)
    return y_test_pca

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.joblib')
    model = joblib.load(model_path)
    return model

def prediction():
    model = load_model()
    X_test_pca, pca_feature_names = load_X_test_pca()
    pred_values = model.predict(X_test_pca)
    return pred_values, pca_feature_names, X_test_pca

### 2.2 Prediction Plots

def plot_pred_dist(y_pred):
    fig, ax = plt.subplots()
    ax.hist(y_pred, bins=[0, 1, 2], align='left', rwidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Goal not Reached (>6 min)', ' Goal reached (<=6 min)'])
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Predictions')
    return fig

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    return fig

def plot_pca_components(X_test_pca, pc_x, pc_y):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=X_test_pca, x=pc_x, y=pc_y, hue='Prediction', palette='Set1', ax=ax)
    ax.set_title(f'PCA Component Plot: {pc_x} vs {pc_y}')
    return fig


### 3 Interpretability
### 3.1 Mean Shap Value Comparison in the Voting Classifier

def plot_shap_across_models():
    mean_abs_shap_comparison = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'mean_abs_shap_comparison.png')
    if os.path.exists(mean_abs_shap_comparison):
        default_plot_image = Image.open(mean_abs_shap_comparison)
        return default_plot_image
    else:
        return None


### 3.2 Mean Shap Value Calculation & Plot

def get_shap_values(model, X, model_type='tree'):
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X)
    else:
        explainer = shap.KernelExplainer(model.predict, X)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer

def plot_shap_values(model_name):
    if model_name == 'XGBoost':
        shap_sum_plot_xgboost = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'shap_sum_plot_xgboost.png')
        if os.path.exists(shap_sum_plot_xgboost):
            plt_xgboost = Image.open(shap_sum_plot_xgboost)
            return plt_xgboost
        else:
            return None
    elif model_name == 'Random Forest':
        shap_sum_plot_rf = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'shap_sum_plot_rf.png')
        if os.path.exists(shap_sum_plot_rf):
            plt_rf = Image.open(shap_sum_plot_rf)
            return plt_rf
        else:
            return None
    elif model_name == 'Logistic Regression':
        shap_sum_plot_logreg = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'shap_sum_plot_logreg.png')
        if os.path.exists(shap_sum_plot_logreg):
            plt_logreg = Image.open(shap_sum_plot_logreg)
            return plt_logreg
        else:
            return None
    else:
        return None

### 3.3 SHAP Violin Plots for for Each Model

def plot_violin_votingclf():
    violin_votingclf = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'shap_violin_plot_votingclf.png')
    if os.path.exists(violin_votingclf):
        violin_votingclf_img = Image.open(violin_votingclf)
        return violin_votingclf_img
    else:
        return None

def plot_shap_violins(model_name):
    if model_name == 'XGBoost':
        shap_violin_xgboost = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'shap_violin_plot_xgboost.png')
        if os.path.exists(shap_violin_xgboost):
            violin_xgboost = Image.open(shap_violin_xgboost)
            return violin_xgboost
        else:
            return None
    elif model_name == 'Random Forest':
        shap_violin_rf = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'shap_violin_plot_rf.png')
        if os.path.exists(shap_violin_rf):
            violin_rf = Image.open(shap_violin_rf)
            return violin_rf
        else:
            return None
    elif model_name == 'Logistic Regression':
        shap_violin_logreg = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'shap_violin_plot_logreg.png')
        if os.path.exists(shap_violin_logreg):
            violin_logreg = Image.open(shap_violin_logreg)
            return violin_logreg
        else:
            return None
    else:
        return None

### 3.4 Local Interpretation for one Observation

# Load SHAP values and explainers once at the start
shap_values_paths = {
    'XGBoost': ('..', '..', 'notebooks', 'shap_values_xgb.pkl', 'explainer_xgb.pkl'),
    'Random Forest': ('..', '..', 'notebooks', 'shap_values_rf.pkl', 'explainer_rf.pkl'),
    'Logistic Regression': ('..', '..', 'notebooks', 'shap_values_logreg.pkl', 'explainer_logreg.pkl')
}

shap_values = {}
explainers = {}

for model_name, paths in shap_values_paths.items():
    shap_values_path = os.path.join(os.path.dirname(__file__), *paths[:3])
    explainer_path = os.path.join(os.path.dirname(__file__), *paths[3:])
    
    if os.path.exists(shap_values_path) and os.path.exists(explainer_path):
        shap_values[model_name] = joblib.load(shap_values_path)
        explainers[model_name] = joblib.load(explainer_path)

def local_interpretation(model_name, X_test_pca, pca_feature_names, index=3):
    if model_name in shap_values and model_name in explainers:
        shap_values_model = shap_values[model_name]
        explainer_model = explainers[model_name]
        
        # Round the values to three decimals
        rounded_shap_values = np.round(shap_values_model[index, :], 3)
        rounded_X_test_pca = np.round(X_test_pca.iloc[index, :], 3)
        
        force_plot = shap.force_plot(
            explainer_model.expected_value, 
            rounded_shap_values, 
            rounded_X_test_pca, 
            feature_names=pca_feature_names, 
            matplotlib=True
        )
        
        return force_plot
    
    elif model_name == 'Combined Voting Classifier':
        if all(m in shap_values and m in explainers for m in ['XGBoost', 'Random Forest', 'Logistic Regression']):
            shap_values_combined = (shap_values['XGBoost'] + shap_values['Random Forest'] + shap_values['Logistic Regression']) / 3
            rounded_shap_values_combined = np.round(shap_values_combined[index, :], 3)
            rounded_X_test_pca = np.round(X_test_pca.iloc[index, :], 3)
            
            force_plot_combined = shap.force_plot(
                explainers['XGBoost'].expected_value, 
                rounded_shap_values_combined, 
                rounded_X_test_pca, 
                feature_names=pca_feature_names, 
                matplotlib=True
            )
            
            return force_plot_combined
    
    return None

### 3.5 Interactive Plot for first 1000 Observations

def interactive_force_plot(model_name, X_test_pca, pca_feature_names, n_observations=1000):
    if model_name in shap_values and model_name in explainers:
        shap_values_model = shap_values[model_name]
        explainer_model = explainers[model_name]
        
        # Round the values to three decimals
        rounded_shap_values = np.round(shap_values_model[:n_observations], 3)
        rounded_X_test_pca = np.round(X_test_pca.iloc[:n_observations], 3)
        
        force_plot = shap.force_plot(
            explainer_model.expected_value, 
            rounded_shap_values, 
            rounded_X_test_pca, 
            feature_names=pca_feature_names
        )
        
        return force_plot
    
    elif model_name == 'Combined Voting Classifier':
        if all(m in shap_values and m in explainers for m in ['XGBoost', 'Random Forest', 'Logistic Regression']):
            shap_values_combined = (shap_values['XGBoost'] + shap_values['Random Forest'] + shap_values['Logistic Regression']) / 3
            rounded_shap_values_combined = np.round(shap_values_combined[:n_observations], 3)
            rounded_X_test_pca = np.round(X_test_pca.iloc[:n_observations], 3)
            
            force_plot_combined = shap.force_plot(
                explainers['XGBoost'].expected_value, 
                rounded_shap_values_combined, 
                rounded_X_test_pca, 
                feature_names=pca_feature_names
            )
            
            return force_plot_combined
    
    return None

# 4 Feature Interpretation
### 4.1 PC Load Reconstruction

### 4.2 PC & Feature Correlation Matrix