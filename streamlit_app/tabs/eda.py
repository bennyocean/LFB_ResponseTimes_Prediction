import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from PIL import Image
from pathlib import Path
import os

image_path = os.path.join(os.path.dirname(__file__), '..', '..', 'img')
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def show_data_exploration():
    
    st.title("Exploratory Data Analysis")

    st.markdown("""
### Data Sets
The exploratory data analysis (EDA) for this project involves four primary datasets, each serving a unique purpose:

1. **LFB Incident Dataset**: This dataset encompasses incidents handled by the London Fire Brigade (LFB) since January 2009. It consists of 39 columns and 716,551 rows, providing a comprehensive record of incident details.
2. **LFB Mobilisation Dataset**: This dataset details the fire engines dispatched to various incidents, comprising 22 columns and 2,373,348 rows. It offers valuable insights into the operational aspects of the LFB's response activities.

3. **Google Maps**: This dataset provides the locations of fire stations across London. It is essential for spatial analysis and understanding the geographic distribution of fire stations.

4. **Bank Holidays Dataset**: This dataset includes all bank holidays in London. It is useful for examining the impact of holidays on incident response times and patterns.
""")
    st.write('Incidents Dataset')
    incidents = pd.read_csv(os.path.join(data_path, 'incidents_prev.csv'))
    incidents['IncidentNumber'] = np.round(incidents['IncidentNumber'])
    mob = pd.read_csv(os.path.join(data_path, 'mob_prev.csv'))
    mob = mob.drop(['IncidentNumber.1'], axis = 1)
    dataset = st.selectbox('Select Dataset', ('Incidents', 'Mobilisation'))

    if dataset == 'Incidents':
        st.write("Incidents")
        st.write(incidents)
    else:
        st.write("Mobilisation")
        st.write(mob)  
    
    st.markdown("""
### Feature Engineering
Feature engineering is a crucial step in transforming raw data into meaningful features that enhance the predictive power of our models. In this analysis, several features were engineered:

1. **From Maps Data**: Calculated the distance of each incident to the nearest fire station. This spatial feature helps in understanding the influence of proximity on response times.

2. **From Geo Data**: Mapped London into a cell grid, enabling a granular analysis of incidents based on geographic regions.

3. **From Bank Holidays**: Created a boolean feature indicating whether an incident occurred on a bank holiday, allowing us to investigate the effect of holidays on incident frequency and response.

4. **From Date Column**: Extracted various temporal features such as whether the incident occurred on a weekend, the time of day, and the day of the week. These features help capture temporal patterns in incident occurrences.

### Setting the Target Variable
The target variable for this analysis is the total response time, calculated as follows:

Since we focus on the first pump arriving at the location of the incident, we use the variable AttendanceTimeSeconds
from the mobilisation dataset. We renamed it TotalResponseTime. 

### Removing Irrelevant Data
To streamline the analysis and enhance model performance, irrelevant data was removed:

1. We focus solely on the first pump arriving at the incident, as subsequent arrivals might introduce noise.
2. Removed several columns not relevant to the target variable, ensuring a leaner and more relevant dataset.
3. Eliminated rows with missing values (NaNs) to maintain data integrity.

### Preparing Data for Modeling
Several preprocessing steps were undertaken to prepare the data for modeling:

1. **Cyclic Encoding**: Applied cyclic encoding to time-related features such as hours of the day and days of the week to ensure that the model correctly interprets these cyclical patterns.
2. **Logarithmic Transformation**: Transformed the numeric feature 'DistanceToStation' and the numeric target variable 'TotalResponseTime' into their logarithmic forms to achieve a more normal distribution, which is beneficial for many machine learning algorithms.

These steps collectively enhance the quality and relevance of the data, setting a solid foundation for subsequent modeling and analysis.

After merging datasets, conducting feature engineering, and removing unnecessary variables, the final dataset comprised 320 columns and 1,537,704 rows.

""")

    df_final_prev = pd.read_csv(os.path.join(data_path, 'df_prev.csv'))
    st.write(df_final_prev)
                
    st.title('Data Visualization')

    image = Image.open(os.path.join(image_path, 'average_total_response_time.png'))
    st.image(image, caption='Average Total Response Time Over the Years', use_column_width=True)
    
    image = Image.open(os.path.join(image_path, 'total_response_time_distribution.png'))
    st.image(image, caption='Total Response Time Distribution', use_column_width=True)
    
    image = Image.open(os.path.join(image_path, 'incident_types_count.png'))
    st.image(image, caption='Incidents Types', use_column_width=True)

    image = Image.open(os.path.join(image_path, 'property_categories_count.png'))
    st.image(image, caption='Property Categories', use_column_width=True)
    
    image = Image.open(os.path.join(image_path, 'bank_holidays_impact.png'))
    st.image(image, caption='Bank Holidays Impact', use_column_width=True)