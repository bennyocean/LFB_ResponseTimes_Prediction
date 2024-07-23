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
The exploratory data analysis (EDA) for this project involves four primary datasets, each serving a unique purpose:

1. **LFB Incident Dataset**: This dataset encompasses incidents handled by the London Fire Brigade (LFB) since January 2009. It consists of 39 columns and 716,551 rows, providing a comprehensive record of incident details.
2. **LFB Mobilisation Dataset**: This dataset details the fire engines dispatched to various incidents, comprising 22 columns and 2,373,348 rows. It offers valuable insights into the operational aspects of the LFB's response activities.

3. **Google Maps**: This dataset provides the locations of fire stations across London. It is essential for spatial analysis and understanding the geographic distribution of fire stations.

4. **Bank Holidays Dataset**: This dataset includes all bank holidays in London. It is useful for examining the impact of holidays on incident response times and patterns.

### Feature Engineering
Feature engineering is a crucial step in transforming raw data into meaningful features that enhance the predictive power of our models. In this analysis, several features were engineered:

1. **From Maps Data**: Calculated the distance of each incident to the nearest fire station. This spatial feature helps in understanding the influence of proximity on response times.

2. **From Geo Data**: Mapped London into a cell grid, enabling a granular analysis of incidents based on geographic regions.

3. **From Bank Holidays**: Created a boolean feature indicating whether an incident occurred on a bank holiday, allowing us to investigate the effect of holidays on incident frequency and response.

4. **From Date Column**: Extracted various temporal features such as whether the incident occurred on a weekend, the time of day, and the day of the week. These features help capture temporal patterns in incident occurrences.

### Calculating the Target Variable
The target variable for this analysis is the total response time, calculated as follows:

1. Converted the 'time of call' and 'time of arrival' to datetime format.
2. Calculated the difference in seconds between the time of call and the time of arrival to obtain the total response time.

### Removing Irrelevant Data
To streamline the analysis and enhance model performance, irrelevant data was removed:

1. Focused solely on the first pump arriving at the incident, as subsequent arrivals might introduce noise.
2. Removed several columns not relevant to the target variable, ensuring a leaner and more relevant dataset.
3. Eliminated rows with missing values (NaNs) to maintain data integrity.
4. After merging datasets, conducting feature engineering, and removing unnecessary variables, the final dataset comprised 320 columns and 1,537,704 rows.

### Preparing Data for Modeling
Several preprocessing steps were undertaken to prepare the data for modeling:

1. **Cyclic Encoding**: Applied cyclic encoding to time-related features such as hours of the day and days of the week to ensure that the model correctly interprets these cyclical patterns.

2. **Logarithmic Transformation**: Transformed the variables 'DistanceToStation' and 'TotalResponseTime' into their logarithmic forms to achieve a more normal distribution, which is beneficial for many machine learning algorithms.

These steps collectively enhance the quality and relevance of the data, setting a solid foundation for subsequent modeling and analysis.
""")


    st.title('Data Visualization')

    image = Image.open(image_path)
    st.image(image, caption='Average Total Response Time Over the Years', use_column_width=True)

