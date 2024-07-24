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
import gdown
import os


st.title("Modelling Storyline")

st.subheader("1. Regression Design")

st.markdown(r"""
                $$
                \text{ResponseTime}_i = \beta_0 + \beta_1 \text{Location}_i + \beta_2 \text{Timing}_i + \beta_3 \text{IncidentType}_i + \epsilon_i
                $$
                """)
    
st.markdown("""
                - _ResponseTime:_ continuous in seconds (ln)
                - _Location:_ distance to station (ln), cell
                - _Timing:_ month, day, hour (ce)
                - _IncidentType:_ incident type, property category
                """)
    
# Data for the chart
data = {
'Design': ['Linear', 'SVR', 'ElasticNet', 'RandomForest', 'XGBoost'],
'R2': [0.253315696, 0.21057167, 0.241756551, 0.240251948, 0.279492873]
}

df = pd.DataFrame(data)

# Create the column chart with a matching background
fig, ax = plt.subplots()
fig.patch.set_facecolor('#0e1117')  # Dark background color to match Streamlit's dark theme
ax.set_facecolor('#0e1117')

# Customize the bar chart
bars = ax.bar(df['Design'], df['R2'], color='#61dafb')

# Set the y-axis label and title with a white color
ax.set_ylabel('R2', color='white')
ax.set_title('Regression Design Comparison', color='white')

# Change the color of the tick labels to white
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Remove the spines
for spine in ax.spines.values():
    spine.set_visible(False)

    # Add data labels inside the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval - 0.05, round(yval, 2), ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')

# Display the chart in the Streamlit app
st.pyplot(fig)


# Create the drop-down menu with the specified options
option = st.selectbox(
    'Select Regression Model',
    ['Linear', 'SVR', 'ElasticNet', 'RandomForest', 'XGBoost'],
    key='regression_model'
)

# Generate DataFrames with specific values for each model
def get_dataframes():
    dataframes = {
        'Linear': pd.DataFrame({
            'MAE': [0.2519433325250212, 0.2519433325250212],
            'MSE': [0.11717495251075341, 0.11717495251075341],
            'RMSE': [0.3423082711690639, 0.3423082711690639],
            'R2': [0.2533156955113437, 0.2533156955113437]
        }, index=['Train', 'Test']),
        'SVR': pd.DataFrame({
            'MAE': [0.2558656794132485, 0.2561106142710682],
            'MSE': [0.12408780585377704, 0.12388264556394395],
            'RMSE': [0.35226099110429054, 0.35196966568717813],
            'R2': [0.2104036650955159, 0.21057167031803015]
        }, index=['Train', 'Test']),
        'ElasticNet': pd.DataFrame({
            'MAE': [0.25533744097237526, 0.25545761314687493],
            'MSE': [0.1193540954470088, 0.11898889477455021],
            'RMSE': [0.34547662069524876, 0.3449476696175091],
            'R2': [0.24052524200604553, 0.24175655092793624]
        }, index=['Train', 'Test']),
        'RandomForest': pd.DataFrame({
            'MAE': [0.09523092764143488, 0.25199285030288776],
            'MSE': [0.01763047607671788, 0.11922500758786787],
            'RMSE': [0.13277980296987144, 0.3452897444000732],
            'R2': [0.887813639728614, 0.2402519483826373]
        }, index=['Train', 'Test']),
        'XGBoost': pd.DataFrame({
            'MAE': [0.2451064377952239, 0.24655474317735396],
            'MSE': [0.1120386260875857, 0.11306704579074599],
            'RMSE': [0.33472171439508625, 0.3362544360908061],
            'R2': [0.28707508430975415, 0.2794928725138395]
        }, index=['Train', 'Test'])
    }
    return dataframes

# Retrieve the DataFrames
dataframes = get_dataframes()

# Display the table for the selected model
st.table(dataframes[option])

st.subheader("2. Baseline Binary Classification")

st.write("LFB Response Time Target: 6 Minutes (LFB, 2022)")

# Dummy data for the pie chart
labels = ["Target Reached (<= 6min, 1)", "Target Not Reached (> 6min, 0)"]
sizes = [0.70625231, 0.29374769] 

# Create a pie chart with black background
background_color = "#0e1117"  # Assuming the background is black
fig, ax = plt.subplots(facecolor=background_color)
ax.set_facecolor(background_color)

wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops=dict(color="white"))
for text in texts:
    text.set_color("white")
for autotext in autotexts:
    autotext.set_color("white")

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Add title
plt.title("LFB Data Target Distribution", color="white")

# Display the plot in Streamlit
st.pyplot(fig)



st.markdown(r"""
                $$
                \text{TargetReached}_i = \beta_0 + \beta_1 \text{Location}_i + \beta_2 \text{Timing}_i + \beta_3 \text{IncidentType}_i + \epsilon_i
                $$
                """)

st.markdown("""
                - _TargetReached:_ binary
                - _Location:_ distance to station (ln), cell
                - _Timing:_ month, day, hour (ce)
                - _IncidentType:_ incident type, property category
                """)




# Data extraction
models = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'XGBoost']
accuracy = [0.67, 0.74, 0.75, 0.76]
recall_class_0 = [0.43, 0.39, 0.29, 0.36]

# Classification reports
classification_reports = {
    'Decision Tree': {
        'Class': ['Class 0', 'Class 1', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
        'Precision': [0.44, 0.77, '', 0.60, 0.67],
        'Recall': [0.43, 0.77, '', 0.60, 0.67],
        'F1-score': [0.44, 0.77, 0.67, 0.60, 0.67],
        'Support': [90243, 217298, 307541, 307541, 307541]
    },
    'Random Forest': {
        'Class': ['Class 0', 'Class 1', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
        'Precision': [0.60, 0.78, '', 0.69, 0.72],
        'Recall': [0.39, 0.89, '', 0.64, 0.74],
        'F1-score': [0.47, 0.83, 0.74, 0.65, 0.72],
        'Support': [90243, 217298, 307541, 307541, 307541]
    },
    'Logistic Regression': {
        'Class': ['Class 0', 'Class 1', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
        'Precision': [0.66, 0.76, '', 0.71, 0.73],
        'Recall': [0.29, 0.94, '', 0.61, 0.75],
        'F1-score': [0.40, 0.84, 0.75, 0.62, 0.71],
        'Support': [90243, 217298, 307541, 307541, 307541]
    },
    'XGBoost': {
        'Class': ['Class 0', 'Class 1', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
        'Precision': [0.67, 0.78, '', 0.73, 0.75],
        'Recall': [0.36, 0.93, '', 0.64, 0.76],
        'F1-score': [0.47, 0.85, 0.76, 0.66, 0.73],
        'Support': [90243, 217298, 307541, 307541, 307541]
    }
}


# Dropdown menu for metric selection
metric = st.selectbox("Select Metric", ["Accuracy", "Recall (Class 0)"])

# Plotting the horizontal bar chart
def plot_bar_chart(metric):
    if metric == "Accuracy":
        values = accuracy
        title = "Baseline Classification Model Comparison"
    elif metric == "Recall (Class 0)":
        values = recall_class_0
        title = "Baseline Classification Model Comparison"
    
    fig, ax = plt.subplots(facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    bars = ax.barh(models, values, color='skyblue')
    ax.set_xlabel(metric, color='white')
    ax.set_title(title, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    for bar in bars:
        width = bar.get_width()
        label_x_pos = width - 0.05
        ax.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center', ha='center', color='black', fontweight='bold')
    
    st.pyplot(fig)

plot_bar_chart(metric)

# Dropdown menu for model selection
model_selected = st.selectbox("Select Model", models, key='model_dropdown')

# Displaying the classification report
def display_classification_report(model):
    report = classification_reports[model]
    df = pd.DataFrame(report)
    st.table(df)

display_classification_report(model_selected)





st.subheader("3. Advanced Binary Classification")

st.write("Employ Principal Component Analysis, Undersampling, Hyperparameter Tuning as well as Bagging and Boosting to simplify calculation execution and improve model performance with respect to recall of the 0-class to minimize risk")

# Define the relative path where the images are stored
image_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'pca')
image_file = os.path.join(image_path, 'pca.png')
image = Image.open(image_file)
st.image(image, use_column_width=True)

st.write("20 Components retain 85% of the explained variance")

