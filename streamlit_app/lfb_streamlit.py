import streamlit as st
import os
from pathlib import Path
from tabs import eda, results, conclusion, model2, home, about

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = Path(__file__).resolve().parent / 'css' / 'style.css'
local_css(css_path)

# Streamlit UI
st.sidebar.title("Table of Contents")
pages = ["Home", "Data", "Model", "Prediction", "Conclusion", "About"]
page = st.sidebar.radio("Go to", pages)
st.sidebar.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
image_path = os.path.join(os.path.dirname(__file__), 'assets', 'logo-datascientest.png')
st.sidebar.image(image_path, use_column_width=True)

# Home
if page == pages[0]:
    home.homepage()
    
# Data
if page == pages[1]:
    eda.show_data_exploration()
    

# Model
if page == pages[2]:
    st.title("Model Selection & Optimization")
    model2.regression_design()
    model2.baseline_binary_classification()
    model2.advanced_binary_classification()


# Prediction
if page == pages[3]:
    st.title("Model Prediction & Interpretability")
    results.load_pred_functions()
    results.load_interpret_functions()
    results.load_pc_load_functions()
    
    
# Conclusion
if page == pages[4]:
    st.title("Conclusion")
    conclusion.conclusion_text()


# About
if page == pages[5]:
    st.title("About")
    about.about()