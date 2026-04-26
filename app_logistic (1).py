import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

# load model
model = pickle.load(open('logistic_model.pkl', 'rb'))

# title
st.title('Telecom customer churn prediction app')

# input variables
SeniorCitizen = st.selectbox('SeniorCitizen', (0, 1))
Partner = st.selectbox('Partner', ('No', 'Yes'))
Dependents = st.selectbox('Dependents', ('No', 'Yes'))
tenure = st.number_input('tenure', min_value=0, max_value=100, value=12)
PhoneService = st.selectbox('PhoneService', ('No', 'Yes'))
MultipleLines = st.selectbox('MultipleLines', ('No', 'Yes', 'No phone service'))
OnlineSecurity = st.selectbox('OnlineSecurity', ('No', 'Yes', 'No internet service'))
OnlineBackup = st.selectbox('OnlineBackup', ('No', 'Yes', 'No internet service'))
DeviceProtection = st.selectbox('DeviceProtection', ('No', 'Yes', 'No internet service'))
TechSupport = st.selectbox('TechSupport', ('No', 'Yes', 'No internet service'))
StreamingTV = st.selectbox('StreamingTV', ('No', 'Yes', 'No internet service'))
StreamingMovies = st.selectbox('StreamingMovies', ('No', 'Yes', 'No internet service'))
PaperlessBilling = st.selectbox('PaperlessBilling', ('No', 'Yes'))
MonthlyCharges = st.number_input('MonthlyCharges', min_value=0.0, max_value=200.0, value=70.0)
TotalCharges = st.number_input('TotalCharges', min_value=0.0, max_value=10000.0, value=1000.0)
gender = st.selectbox('gender', ('Female', 'Male'))
InternetService = st.selectbox('InternetService', ('DSL', 'Fiber optic', 'No'))
Contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
PaymentMethod = st.selectbox(
    'PaymentMethod',
    ('Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check')
)

# encoding
binary_map = {'No': 0, 'Yes': 1}

Partner = binary_map[Partner]
Dependents = binary_map[Dependents]
PhoneService = binary_map[PhoneService]
PaperlessBilling = binary_map[PaperlessBilling]

service_map = {'No': 0, 'Yes': 1, 'No internet service': 0, 'No phone service': 0}
MultipleLines = service_map[MultipleLines]
OnlineSecurity = service_map[OnlineSecurity]
OnlineBackup = service_map[OnlineBackup]
DeviceProtection = service_map[DeviceProtection]
TechSupport = service_map[TechSupport]
StreamingTV = service_map[StreamingTV]
StreamingMovies = service_map[StreamingMovies]

gender_Male = 1 if gender == 'Male' else 0
InternetService_Fiber_optic = 1 if InternetService == 'Fiber optic' else 0
InternetService_No = 1 if InternetService == 'No' else 0
Contract_One_year = 1 if Contract == 'One year' else 0
Contract_Two_year = 1 if Contract == 'Two year' else 0
PaymentMethod_Credit_card_automatic = 1 if PaymentMethod == 'Credit card (automatic)' else 0
PaymentMethod_Electronic_check = 1 if PaymentMethod == 'Electronic check' else 0
PaymentMethod_Mailed_check = 1 if PaymentMethod == 'Mailed check' else 0

# create a dataframe
input_features = pd.DataFrame({
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'DeviceProtection': [DeviceProtection],
    'TechSupport': [TechSupport],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'PaperlessBilling': [PaperlessBilling],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges],
    'gender_Male': [gender_Male],
    'InternetService_Fiber optic': [InternetService_Fiber_optic],
    'InternetService_No': [InternetService_No],
    'Contract_One year': [Contract_One_year],
    'Contract_Two year': [Contract_Two_year],
    'PaymentMethod_Credit card (automatic)': [PaymentMethod_Credit_card_automatic],
    'PaymentMethod_Electronic check': [PaymentMethod_Electronic_check],
    'PaymentMethod_Mailed check': [PaymentMethod_Mailed_check]
})

# scaling based on training notebook preprocessing
train_data = pd.read_csv('telecom_churn_data.csv')
train_data['TotalCharges'] = train_data['TotalCharges'].replace(' ', pd.NA)
train_data['TotalCharges'] = pd.to_numeric(train_data['TotalCharges'], errors='coerce')
for c in train_data.select_dtypes(include=['object']).columns:
    train_data[c] = train_data[c].fillna(train_data[c].mode()[0])
for c in train_data.select_dtypes(include=['int64', 'float64']).columns:
    train_data[c] = train_data[c].fillna(train_data[c].median())

scaler_tenure = StandardScaler()
scaler_monthly = StandardScaler()
scaler_total = StandardScaler()
scaler_tenure.fit(train_data[['tenure']])
scaler_monthly.fit(train_data[['MonthlyCharges']])
scaler_total.fit(train_data[['TotalCharges']])

input_features[['tenure']] = scaler_tenure.transform(input_features[['tenure']])
input_features[['MonthlyCharges']] = scaler_monthly.transform(input_features[['MonthlyCharges']])
input_features[['TotalCharges']] = scaler_total.transform(input_features[['TotalCharges']])

if st.button('Predict'):
    predictions = model.predict(input_features)
    if predictions[0] == 1:
        st.error('⚠️Customer is likely to churn')
    else:
        st.success('😊Customer is likely to stay')
