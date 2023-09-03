import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler

# Load the pre-trained model
model_filename = 'Random_Forest_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

sc_X_filename = 'fitted_scaler.pkl'
with open(sc_X_filename, 'rb') as file:
    sc_X = pickle.load(file)

# Streamlit app
def main():
    st.title('Diabetes Outcome Prediction')
    st.markdown("Enter the patient's details to predict diabetes outcome.")

    # Create input fields for user
    col1, col2 = st.columns(2)

    #with col1:
    #    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20)
    #    glucose = st.number_input('Glucose', min_value=100, max_value=200)
    #    blood_pressure = st.number_input('Blood Pressure', min_value=70, max_value=150)
    #    skin_thickness = st.number_input('Skin Thickness', min_value=20, max_value=100)

    #with col2:
    #    insulin = st.number_input('Insulin',min_value=100, max_value=900)
    #    bmi = st.number_input('BMI', min_value=25.0, max_value=60.0)
    #    diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.5, max_value=2.0)
    #    age = st.number_input('Age', min_value=30, max_value=100)
    
    with col1:
        pregnancies = st.number_input('Pregnancies',min_value=0)
        glucose = st.number_input('Glucose',min_value=0)
        blood_pressure = st.number_input('Blood Pressure',min_value=0)
        skin_thickness = st.number_input('Skin Thickness',min_value=0)

    with col2:
        insulin = st.number_input('Insulin',min_value=0)
        bmi = st.number_input('BMI',min_value=0.0)
        diabetes_pedigree = st.number_input('Diabetes Pedigree Function',min_value=0.0)
        age = st.number_input('Age', min_value=0)

    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })

    input_data = input_data.drop(columns = ['SkinThickness','Insulin'])

    # Preprocess the input data
    input_data_scaled = sc_X.transform(input_data)
    prediction = model.predict(input_data_scaled)

    if st.button('Predict'):
        # Use the model to make predictions
        prediction_text = 'patient has diabetes' if prediction[0] == 1 else 'Patient has not diabetes'
        prediction_color = 'red' if prediction[0] == 1 else 'green'
        st.markdown('## Prediction Result')
        prediction_result = st.empty()
        with prediction_result:
            if prediction[0] == 1:
                st.markdown('<div style="background-color:#ffcccc; padding:10px; border-radius:10px;">'
                            f'<h4 style="color:red;">{prediction_text}</h4>'
                            '</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div style="background-color:#dcffdc; padding:10px; border-radius:10px;">'
                            f'<h4 style="color:green;">{prediction_text}</h4>'
                            '</div>',
                            unsafe_allow_html=True)
                
if __name__ == '__main__':
    main()
