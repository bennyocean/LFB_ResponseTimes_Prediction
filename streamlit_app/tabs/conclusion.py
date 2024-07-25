import streamlit as st

def conclusion_text():
    # Summary of the Project
    st.subheader("Summary")
    st.write("""
    - We ensure data quality and consistency through cleaning, handling missing values, encoding categorical variables, and scaling features.
    - Our EDA helps understand data distribution, identify patterns, and provides insights into key factors influencing response times.
    - Furthermore, we develop and evaluate a binary classification model with 23 principal components to predict response time classes using the Voting Classifier model.
    - The Voting Classifier model demonstrates good performance but also highlights areas for improvement, such as a high number of false negatives. 
    """)

    # Limitations
    st.subheader("Limitations & Future Opportunities")
    st.write("""
    Athough we encountered various constraints (e.g., limited computing power, project time constraints, and lack of domain knowledge), future projects can enhance model performance by focusing on the following aspects:
    - Incorporate additional features, such as data on major events in London, live-traffic information, and seasonal weather patterns.
    - Review and integrate findings from scientific research related to using state-of-the-art machine learning models and consulting literature on emergency response optimization.
    - Focus on adjusting features based on PCA and SHAP value analysis to enhance model performance and interpretability.
    """)

    # Key Findings
    st.subheader("Contribution")
    st.write("""
    Finally, the London Fire Brigade benefits manifold from our findings: 
    - Gaining insights into influential features like "DistanceStationLog" and "Hour_cos" optimizes strategic resource allocation.
    - Identifying key factors allows for targeted improvements in response times and data-driven decision-making processes boosts operational efficiency.
    - Integrating datetime and geographic data improves predictive capabilities and operational readiness.
    """)
