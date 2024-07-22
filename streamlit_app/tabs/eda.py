import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from PIL import Image
from pathlib import Path
import os

image_path = os.path.join(os.path.dirname(__file__), '..', '..', 'img', 'average_total_response_time.png')

def show_data_exploration():
    
    st.title("Exploratory Data Analysis")

    st.markdown("""
                ### Data Sets
                - **LFB Incident Dataset**: incidents handled since January 2009
                    - *39 columns, 716.551 rows*
                - **LFB Mobilisation Dataset**: fire engines dispatched to incidents
                    - *22 columns, 2.373.348 rows*
                - **Google Maps**: locations of fire stations in London
                - **Bank Holidays Dataset**: all bank holidays in London
                
                ### Feature Engineering
                - **From maps data**: the distance of the incident to the nearest fire station
                - **From geo data**: mapped London into a cell grid
                - **From bank holidays**: created a boolean feature indicating if the incident happened on a holiday
                - **From date column**: extracted if the incident occurred on a weekend, time of day, and weekday
                
                ### Calculate Target Variable
                - Converted time of call and time of arrival to datetime format
                - Calculated the difference in seconds to get the total response time
                
                ### Removing Irrelevant Data
                - Focused only on the first pump arriving at the incident
                - Removed several columns not relevant to the target variable
                - Removed NaNs
                - After merging datasets, feature engineering, and removing variables:
                    - *320 columns, 1.537.704 rows*
                    
                ### Preparing Data for Modeling
                - Used cyclic encoding for the time-related features to ensure correct interpretation by the model
                - Transformed the variables *DistanceToStation* and *TotalResponseTime* into logarithmic form to ensure a normal distribution
    """)


    st.title('Data Visualization')

    image = Image.open(image_path)
    st.image(image, caption='Average Total Response Time Over the Years', use_column_width=True)

