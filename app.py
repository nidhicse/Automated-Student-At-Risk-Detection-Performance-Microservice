import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Student At-Risk Detection", page_icon="🎓", layout="centered")

st.title("🎓 Student At-Risk Detection Dashboard")
st.markdown("Predict the likelihood of a student being at-risk based on demographic and academic profiles.")

# Form inputs
with st.form("prediction_form"):
    st.subheader("Demographic & Academic Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["female", "male"])
        race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
        parental_level_of_education = st.selectbox("Parental Education", [
            "some high school", 
            "high school", 
            "some college", 
            "associate's degree", 
            "bachelor's degree", 
            "master's degree"
        ])
        
    with col2:
        lunch = st.selectbox("Lunch Plan", ["standard", "free/reduced"])
        test_preparation_course = st.selectbox("Test Prep Course", ["none", "completed"])
        
    st.subheader("Academic Scores")
    math_score = st.slider("Math Score", 0, 100, 50)
    reading_score = st.slider("Reading Score", 0, 100, 50)
    writing_score = st.slider("Writing Score", 0, 100, 50)
    
    submitted = st.form_submit_button("Predict Risk")

if submitted:
    with st.spinner("Analyzing student profile and calculating SHAP values..."):
        try:
            # Prepare data
            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                math_score=math_score,
                reading_score=reading_score,
                writing_score=writing_score
            )
            
            pred_df = data.get_data_as_data_frame()
            
            # Predict
            pipeline = PredictPipeline()
            results = pipeline.predict(pred_df)
            
            probability = results["at_risk_probability"]
            fig = results["shap_plot"]
            
            st.divider()
            
            # Display results
            if probability > 0.5:
                st.error(f"⚠️ **High Risk Alert:** The student has a **{probability*100:.1f}%** probability of being at-risk.")
            else:
                st.success(f"✅ **Low Risk:** The student has a **{probability*100:.1f}%** probability of being at-risk.")
                
            st.subheader("Decision Explainer (SHAP)")
            st.markdown("This chart explains how each feature influenced the prediction. **Red bars** push the risk higher, while **blue bars** push the risk lower.")
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
