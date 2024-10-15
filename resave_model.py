import joblib
import os

# Path to your existing model file (update this to where your current model is located)
current_model_path = '/Users/bennyschellinger/Desktop/DataScience/Github/DataScientest_LfB_DataScience/streamlit_app/model.joblib'

# Load the existing model
print(f"Loading the model from {current_model_path}...")
model = joblib.load(current_model_path)

# Create a 'models' directory if it doesn't exist
new_model_dir = 'models'
if not os.path.exists(new_model_dir):
    os.makedirs(new_model_dir)

# Define the new model path where you want to save it
new_model_path = os.path.join(new_model_dir, 'vclf_hard_grid_resaved.pkl')

# Re-save the model to the new path
print(f"Saving the model to {new_model_path}...")
joblib.dump(model, new_model_path)

print("Model saved successfully!")
