import pandas as pd
import numpy as np
import streamlit as st
import pickle
st.title("İşe Alınma Tahmin Uygulaması")
age = st.number_input('Yaş', min_value=18, max_value=65, value=30)
education = st.selectbox('Eğitim Seviyesi', ['Önlisans', 'Lisans', 'Yüksek Lisans', 'Doktora'])
experience = st.slider('Deneyim (Yıl)', 0, 40, 5)
distance = st.slider('Şirketten Uzaklık (km)', 0, 100, 10)
gender = st.selectbox('Cinsiyet', ['Erkek', 'Kadın'])

education_mapping = {'Önlisans': 1, 'Lisans': 2, 'Yüksek Lisans': 3, 'Doktora': 4}
gender_mapping = {'Erkek': 0, 'Kadın': 1}

education_num = education_mapping[education]
gender_num = gender_mapping[gender]

user_data = {
            'Age': age, 
            'EducationLevel': education_num, 
            'ExperienceYears': experience,
            'DistanceFromCompany': distance,
            'Gender': gender_num
        }
        
features = pd.DataFrame(user_data, index=[0])
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)    
    
#columns_needed = loaded_model.feature_names_in_
#user_input = user_input.reindex(columns=columns_needed, fill_value=0)
    
    prediction = loaded_model.predict(user_input)
    
    # Tahmin sonucunu gösterme
    st.subheader('Tahmin Sonucu')
    st.write('İşe Alınma Durumu: {}'.format('Alınacak' if prediction[0] == 1 else 'Alınmayacak'))

