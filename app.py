import streamlit as st
import pandas as pd
from PIL import Image
import pickle

# Load the trained pipeline
model = pickle.load(open('promotion_pipeline.pkl', 'rb'))

#page setup
st.set_page_config(page_title="EMPLOYEE PROMOTION PREDICTION", layout="centered")

# Background image CSS
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/free-vector/office-interior-open-space-flat-design-vector-illustration-businessmen-talking-modern-meeting-room-conference-hall_126523-3102.jpg?semt=ais_hybrid&w=740&q=80");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.7);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("Employee Promotion Eligibility Predictor")

st.write("Enter employee details below to check if they are eligible for promotion:")

# Input fields
department = st.selectbox("Department", ["HR", "Finance", "IT", "Operations", "Marketing"])
job_role = st.selectbox("Job Role", ["Manager", "Engineer", "Analyst", "Executive", "Intern"])

performance_score = st.number_input("Performance Score", 0.0, 100.0, step=0.1)
kpi_score = st.number_input("KPI Score", 0.0, 100.0, step=0.1)
attendance = st.number_input("Attendance (%)", 0.0, 100.0, step=0.1)
peer_rating = st.number_input("Peer Rating", 0.0, 10.0, step=0.1)
task_completion = st.number_input("Task Completion (%)", 0.0, 100.0, step=0.1)



# Combine inputs into a dataframe
input_data = pd.DataFrame([{
    'Department': department,
    'Job Role': job_role,
    'Performance Score': performance_score,
    'KPI Score': kpi_score,
    'Attendance (%)': attendance,
    'Peer Rating': peer_rating,
    'Task Completion (%)': task_completion,
   
}])

# Prediction button
if st.button(" Predict Promotion Eligibility"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("Employee is Eligible for Promotion!")
    else:
        st.error("Employee is Not Eligible for Promotion.")