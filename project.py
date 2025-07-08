#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Load the model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title="No-show Risk Predictor", layout="centered")
st.title("🩺 No-show Prediction & Intervention Recommendation")
st.markdown("Predict the risk of a patient missing their appointment and suggest appropriate actions.")

st.sidebar.header("📋 Patient & Appointment Details")

def user_input_features():
    age = st.sidebar.slider('Age', 0, 115, 30)
    scholarship = st.sidebar.selectbox('Scholarship', [0, 1])
    hipertension = st.sidebar.selectbox('Hypertension', [0, 1])
    diabetes = st.sidebar.selectbox('Diabetes', [0, 1])
    alcoholism = st.sidebar.selectbox('Alcoholism', [0, 1])
    handcap = st.sidebar.selectbox('Handicap', [0, 1, 2, 3, 4])
    sms_received = st.sidebar.selectbox('SMS Received', [0, 1])
    lead_time = st.sidebar.slider('Lead Time (days)', 0, 60, 5)

    data = {
        'Age': age,
        'Scholarship': scholarship,
        'Hipertension': hipertension,
        'Diabetes': diabetes,
        'Alcoholism': alcoholism,
        'Handcap': handcap,
        'SMS_received': sms_received,
        'LeadTime': lead_time
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

st.subheader("📄 Entered Information")
st.dataframe(df_input)

if st.button("🔍 Predict No-show Risk"):
    input_data = df_input[[ 'Age', 'Scholarship', 'Hipertension',
                           'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'LeadTime']]

    # Predict
    risk_score = model.predict_proba(input_data)[:, 1][0]

    # Risk thresholds
    high_risk_threshold = 0.7
    medium_risk_threshold = 0.4

    if risk_score >= high_risk_threshold:
        risk_level = '🔴 High Risk'
        suggested_action = 'Call the patient or offer rescheduling options.'
        gauge_color = "red"
    elif risk_score >= medium_risk_threshold:
        risk_level = '🟠 Medium Risk'
        suggested_action = 'Send an extra SMS reminder.'
        gauge_color = "orange"
    else:
        risk_level = '🟢 Low Risk'
        suggested_action = 'Standard reminder is sufficient.'
        gauge_color = "green"

    # Gauge plot
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        title={'text': "No-show Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "tomato"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Summary Panel
    st.markdown("### 🧠 Prediction Summary")
    st.success(f"**Risk Level**: {risk_level}")
    st.info(f"**Recommended Action**: {suggested_action}")
    st.write(f"**Predicted No-show Probability**: `{risk_score:.4f}`")



# In[ ]:




