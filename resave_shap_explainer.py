import joblib
import os

# Paths to your existing SHAP explainer and SHAP values (update these if different)
xgb_explainer_path = os.path.join('notebooks', 'explainer_xgb.pkl')
xgb_shap_values_path = os.path.join('notebooks', 'shap_values_xgb.pkl')

# Load the existing SHAP explainer and SHAP values
print(f"Loading XGBoost SHAP explainer from {xgb_explainer_path}...")
xgb_explainer = joblib.load(xgb_explainer_path)

print(f"Loading XGBoost SHAP values from {xgb_shap_values_path}...")
xgb_shap_values = joblib.load(xgb_shap_values_path)

# Create a 'shap_models' directory if it doesn't exist
new_shap_model_dir = 'shap_models'
if not os.path.exists(new_shap_model_dir):
    os.makedirs(new_shap_model_dir)

# Define the new paths for saving
new_xgb_explainer_path = os.path.join(new_shap_model_dir, 'explainer_xgb_resaved.pkl')
new_xgb_shap_values_path = os.path.join(new_shap_model_dir, 'shap_values_xgb_resaved.pkl')

# Re-save the SHAP explainer and SHAP values
print(f"Saving the XGBoost SHAP explainer to {new_xgb_explainer_path}...")
joblib.dump(xgb_explainer, new_xgb_explainer_path)

print(f"Saving the XGBoost SHAP values to {new_xgb_shap_values_path}...")
joblib.dump(xgb_shap_values, new_xgb_shap_values_path)

print("SHAP explainer and values saved successfully!")
