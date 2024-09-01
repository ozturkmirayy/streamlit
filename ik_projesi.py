import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def train_model():
    try:
        # Veriyi okuma
        data = pd.read_csv('recruitment_data.csv')
        
        # Eksik değerleri kontrol etme ve doldurma
        if data.isnull().sum().sum() > 0:
            data = data.fillna(0)
        
        print("Veri sütunları:", data.columns)

        # Özellikler ve hedef değişkeni ayırma
        X = data.drop('HiringDecision', axis=1)
        y = data['HiringDecision']
        
        # Veriyi eğitim ve test setlerine ayırma
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Modeli eğitme
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Tahmin ve doğruluk
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Doğruluğu: {accuracy:.2f}")
        
        # Modeli kaydetme
        with open('model.pkl', 'wb') as file:
            pickle.dump(model, file)

    except Exception as e:
        print(f"Model eğitimi sırasında bir hata oluştu: {e}")

# train_model()  # Bu satırı yorum satırına alın, yalnızca model eğitimine ihtiyacınız olduğunda çalıştırın.

st.title("İşe Alınma Tahmin Uygulaması")

def get_user_input():
    try:
        # Kullanıcı girdileri
        age = st.number_input('Yaş', min_value=18, max_value=65, value=30)
        education = st.selectbox('Eğitim Seviyesi', ['Önlisans', 'Lisans', 'Yüksek Lisans', 'Doktora'])
        experience = st.slider('Deneyim (Yıl)', 0, 40, 5)
        distance = st.slider('Şirketten Uzaklık (km)', 0, 100, 10)
        gender = st.selectbox('Cinsiyet', ['Erkek', 'Kadın'])

        # Haritalamalar
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
        return features
    
    except Exception as e:
        st.error(f"Girdi işlemleri sırasında bir hata oluştu: {e}")
        return pd.DataFrame()  # Boş bir DataFrame döner

user_input = get_user_input()

try:
    # Eğitilmiş modeli yükleme
    with open('model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    
    # Girdileri modelin beklediği özelliklerle hizalama
    columns_needed = loaded_model.feature_names_in_
    user_input = user_input.reindex(columns=columns_needed, fill_value=0)
    
    # Tahmin yapma
    prediction = loaded_model.predict(user_input)
    
    # Tahmin sonucunu gösterme
    st.subheader('Tahmin Sonucu')
    st.write('İşe Alınma Durumu: {}'.format('Alınacak' if prediction[0] == 1 else 'Alınmayacak'))

except FileNotFoundError:
    st.error("Model dosyası bulunamadı, lütfen modeli eğitip tekrar deneyin.")
except NameError as ne:
    st.error(f"Bir NameError oluştu: {ne}")
except Exception as e:
    st.error(f"Tahmin işlemi sırasında bir hata oluştu: {e}")
