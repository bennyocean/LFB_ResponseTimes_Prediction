import streamlit as st

def introduction():
    st.title("London Fire Brigade Response Time")
    st.write("""
             In this project we aim to analyze and predict response times for the London Fire Brigade (LFB). 
             The LFB is the busiest fire and rescue service in the UK and requires precise and rapid responses to emergencies. 
             This project aims to predict response times using mobilization and incident data, with the goal of 
             optimizing the LFB's operational efficiency, resource allocation, and overall effectiveness in emergency response.
             """)
    
    st.subheader("Project Team")
    st.write("Clemens Paulsen, Dr. Benjamin Schellinger and Ismarah Maier")
    
    # Was wollen wir genau erreichen? 