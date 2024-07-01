import numpy as np
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

model = pickle.load(open('linear_regression_model.pkl', 'rb'))

with st.sidebar:
    selected = option_menu("Choose Prediction System", ['Salary_Prediction', 'Heart_Prediction'])

if selected == 'Salary_Prediction':
    st.title('Salary Prediction using ML')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age', 0, 100)
        leaves_used = st.number_input('Leaves Used', 0, 200)
        leaves_remaining = st.number_input('Leaves Remaining', 0, 200)

    with col2:
        ratings = st.number_input('Ratings', 0, 10)
        sex = st.selectbox('Sex', ['F', 'M'])
        designation = st.selectbox('Designation', ['Analyst', 'Associate', 'Director', 'Manager', 'Senior Analyst', 'Senior Manager'])

    with col3:
        unit = st.selectbox('Unit', ['Finance', 'IT', 'Management', 'Marketing', 'Operations', 'Web'])
        past_exp = st.number_input('Past Exp', 0, 200)

    if st.button('Predict'):
        # Convert categorical variables to one-hot encoding
        sex_encoded = 1 if sex == 'F' else 0
        designation_encoded = [1 if designation == d else 0 for d in ['Analyst', 'Associate', 'Director', 'Manager', 'Senior Analyst', 'Senior Manager']]
        unit_encoded = [1 if unit == u else 0 for u in ['Finance', 'IT', 'Management', 'Marketing', 'Operations', 'Web']]

        data = {
            'AGE': [age],
            'LEAVES USED': [leaves_used],
            'LEAVES REMAINING': [leaves_remaining],
            'RATINGS': [ratings],
            'PAST EXP': [past_exp],
            'SEX_F': [sex_encoded],
            'SEX_M': [1 - sex_encoded],
            'DESIGNATION_Analyst': [designation_encoded[0]],
            'DESIGNATION_Associate': [designation_encoded[1]],
            'DESIGNATION_Director': [designation_encoded[2]],
            'DESIGNATION_Manager': [designation_encoded[3]],
            'DESIGNATION_Senior Analyst': [designation_encoded[4]],
            'DESIGNATION_Senior Manager': [designation_encoded[5]],
            'UNIT_Finance': [unit_encoded[0]],
            'UNIT_IT': [unit_encoded[1]],
            'UNIT_Management': [unit_encoded[2]],
            'UNIT_Marketing': [unit_encoded[3]],
            'UNIT_Operations': [unit_encoded[4]],
            'UNIT_Web': [unit_encoded[5]]
        }

        # Create a DataFrame from the form data
        df = pd.DataFrame(data)

        # Make prediction
        prediction = model.predict(df)

        # Return the prediction
        st.write(f"Predicted Salary: ${prediction[0]:,.2f}")

if selected == 'Heart_Prediction':
    st.title('Heart Prediction using ML')
