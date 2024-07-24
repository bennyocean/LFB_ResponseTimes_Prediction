import streamlit as st

def conclusion_text():
    # Summary of the Project
    st.subheader("Summary of the Project")
    st.write("""
    - We ensure data quality and consistency through cleaning, handling missing values, encoding categorical variables, and scaling features.
    - Our EDA helps understand data distribution, identify patterns, and provides insights into key factors influencing response times.
    - Furthermore, we develop and evaluate a binary classification model with 23 principal components to predict response time classes using the Voting Classifier model.
    - The Voting Classifier model demonstrates good performance but also highlights areas for improvement, such as a high number of false negatives. 
    """)

    # Limitations
    st.subheader("Limitations & Future Opportunities")
    st.write("""
    - **Limited computing power**, affecting the complexity and scale of the models we could use.
    - **Time constraints**, restricting the depth of our analysis and experimentation with different models.
    - **Lack of domain knowledge**, which may have impacted the feature selection and interpretation of results.
    """)

    # Future Opportunities
    st.write("""
    However, future projects can enhance model performance by focusing on the following aspects:
    - Incorporate additional features, such as data on major events in London, live-traffic information, and seasonal weather patterns.
    - Review and integrate findings from scientific research related to using state-of-the-art machine learning models and consulting literature on emergency response optimization.
    - Focus on adjusting features based on PCA and SHAP value analysis to enhance model performance and interpretability.
    """)

    # Key Findings
    st.subheader("Contribution to the London Fire Brigade")
    st.write("""
    The London Fire Brigade benefits manifold from our findings: 
    - Gaining insights into influential features like "DistanceStationLog" and "Hour_cos" optimizes strategic resource allocation.
    - Identifying key factors allows for targeted improvements in response times and data-driven decision-making processes boosts operational efficiency.
    - Integrating datetime and geographic data improves predictive capabilities and operational readiness.
    """)

    # Conclusion

    st.write("""
    Concluding, this project provides valuable insights into the factors influencing response times, offering several actionable opportunities for future enhancements. 
    By leveraging additional data sources and refining models, the London Fire Brigade can make more informed decisions, ultimately enhance business operations and foremost better serving the community.
    """)
