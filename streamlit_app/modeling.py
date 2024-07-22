import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

df = pd.read_parquet('../data/df_cleaned_for_classification_models.parquet')
df.head() 