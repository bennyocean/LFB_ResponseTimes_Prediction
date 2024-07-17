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

df = pd.read_parquet('../data/df_cleaned_for_classification_models.parquet')
X = df.drop("ResponseTimeBinary", axis = 1)
y = df["ResponseTimeBinary"]
pca = PCA(n_components=0.85)
X = pca.fit_transform(X)
rUs = RandomUnderSampler(random_state=666)
X, y = rUs.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 666)

st.title("London Fire Brigade : binary classification project")
st.sidebar.title("Table of contents")
pages=["Exploration", "DataVizualization", "Modelling"]
page=st.sidebar.radio("Go to", pages)

# Data exploration
if page == pages[0] : 
    st.write("### Presentation of data")
    st.dataframe(df.head(10))
    st.write(df.shape)
    st.dataframe(df.describe())
    if st.checkbox("Show NA") :
        st.dataframe(df.isna().sum())

# Data Viz        
if page == pages[1] : 
    st.write("### DataVizualization")
    df.Age.hist()

# Modelling    
if page == pages[2] : 
    st.write("### Modelling")
    
    model = joblib.load('../Pickles/vclf_hard_grid.pkl')
    xgboost = model.named_estimators_['XGboost']
    rf = model.named_estimators_['RF']
    logreg = model.named_estimators_['LogReg']
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    X = df.drop("ResponseTimeBinary", axis=1)

    # Fit PCA
    pca = PCA(n_components=0.85)
    pca.fit(X)

    # Get the loadings
    loadings = pca.components_.T  # Transpose to get the correct shape

    # Create a DataFrame for easier interpretation
    loadings_df = pd.DataFrame(loadings, index=X.columns, columns=[f'PC{i+1}' for i in range(loadings.shape[1])])

    # Display the loadings for PC6
    pc6_loadings = loadings_df['PC6'].sort_values(ascending=False)
    print(pc6_loadings)

    # Plot the loadings for PC6
    top_n = 10  # Number of top and bottom features to display
    pc6_loadings_top_bottom = pd.concat([pc6_loadings.head(top_n), pc6_loadings.tail(top_n)])

    plt.figure(figsize=(10, 6))
    sns.barplot(x=pc6_loadings_top_bottom.values, y=pc6_loadings_top_bottom.index)
    plt.title('Top and Bottom PCA Loadings for PC6')
    plt.xlabel('Loading Value')
    plt.ylabel('Original Feature')
    plt.show()
    
    


    
   




    
