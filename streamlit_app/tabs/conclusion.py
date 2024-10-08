import streamlit as st

def conclusion_text():
    # Summary of the Project
    st.subheader("Summary")
    st.write("""
            - Our EDA provided insights into key factors influencing response times and helped in selecting and engineering features.
            - The model optimization process yielded a 25 percentage point performance increase compared to our best-performing baseline model.
            - PC6 and PC3, driven by datetime and geographical variables, were identified as the most influential features.
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
