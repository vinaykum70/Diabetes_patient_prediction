import numpy as np
import streamlit as st
from pickle import load

model = load(open('models/Diabetes_final_lr.pkl', 'rb'))

scaler = load(open('models/StandardScaler.pkl','rb'))

st.title('Diabetes Prediction Web App')

Pregnancies = st.text_input('Number of Pregnancies', placeholder = 'Enter number of Pregnancies')
Glucose = st.text_input('Glucose Level', placeholder='Enter Glucose level' )
BloodPressure = st.text_input('Blood Pressure value', placeholder='Enter Blood Pressure value')
SkinThickness = st.text_input('Skin Thickness value',placeholder ='Enter Skin Thikness value')
Insulin = st.text_input('Insulin level',placeholder='Enter Insulin level')
BMI = st.text_input('BMI value',placeholder='Enter BMI value')
DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value',placeholder='Enter Diabetes Pedigree Function value')
Age = st.text_input('Age of the Person', placeholder='Enter Age of the Person')

btn_click = st.button('Diabetes Test Result')

if btn_click==True:
    if Pregnancies and Glucose and BloodPressure and SkinThickness and Insulin and BMI and DiabetesPedigreeFunction and Age:
        query_point = np.array([int(Pregnancies),int(Glucose),int(BloodPressure),int(SkinThickness), int(Insulin),float(BMI),float(DiabetesPedigreeFunction),int(Age)]).reshape(1,-1)
        query_point_transformed = scaler.transform(query_point)

        prediction = model.predict(query_point_transformed)

        if(prediction==0):
            st.success("The person is not diabetic")
        else:
            st.success("The person is diabetic")

        


