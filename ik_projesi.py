import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Modeli eğitme fonksiyonu
def train_model():
    data = pd.read_csv('recruitment_data.csv')
    print(data.columns)

    X = data.drop('HiringDecision', axis=1)
    y = data['HiringDecision']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Doğruluğu: {accuracy}")

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Eğer model yoksa bir defaya mahsus oluşturun ve kaydedin
train_model()

# Streamlit sayfa ayarları
st.set_page_config(
    page_title="İşe Alım Tahminleme",
    page_icon="https://www.cottgroup.com/images/Zoo/gorsel/insan-kaynaklari-analitigi-ic-gorsel-2.webp",
)

# Açıklama metinleri ve görseller
st.markdown("Merhaba! Bu uygulama, adayların işe alım sürecinde başarıyla değerlendirilip değerlendirilemeyeceğini öngörmek için geliştirilmiştir.")
st.image("https://www.cottgroup.com/images/Zoo/gorsel/insan-kaynaklari-analitigi-ic-gorsel-2.webp")
st.markdown("Bu uygulama, geçmiş verilerden elde edilen bir makine öğrenmesi modeli kullanarak tahminler yapmaktadır. Modele girilen bilgiler doğrultusunda, bir adayın işe alınma ihtimali “Alınacak” veya “Alınmayacak” olarak belirlenir.")
st.image("https://www.cottgroup.com/images/Zoo/gorsel/insan-kaynaklari-analitigi-ic-gorsel-1.webp")
st.markdown("Önemli Not: Bu uygulama yalnızca bilgilendirme amaçlıdır ve işe alım kararları verirken tek başına kullanılmamalıdır.")

# Kullanıcı girişi fonksiyonu
def get_user_input():
    age = st.number_input('Yaş', min_value=18, max_value=65, value=30)
    education = st.selectbox('Eğitim Seviyesi', ['Önlisans', 'Lisans', 'Yüksek Lisans', 'Doktora'])
    experience = st.slider('Deneyim (Yıl)', 0, 40, 5)
    distance = st.slider('Şirketten Uzaklık (km)', 0, 100, 10)
    gender = st.selectbox('Cinsiyet', ['Erkek', 'Kadın'])

    # Eğitim seviyesi ve cinsiyet için sayısal değerler
    education_mapping = {'Önlisans': 1, 'Lisans': 2, 'Yüksek Lisans': 3, 'Doktora': 4}
    gender_mapping = {'Erkek': 0, 'Kadın': 1}

    education_num = education_mapping[education]
    gender_num = gender_mapping[gender]

    # Kullanıcı verilerini DataFrame olarak oluşturma
    user_data = {
        'Age': age, 
        'EducationLevel': education_num, 
        'ExperienceYears': experience,
        'DistanceFromCompany': distance,
        'Gender': gender_num
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

# Kullanıcı girdisini alma
user_input = get_user_input()

# Modeli yükleme
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Kullanıcı girdisini modelin ihtiyaç duyduğu sütun düzenine göre yeniden düzenleme
columns_needed = loaded_model.feature_names_in_
user_input = user_input.reindex(columns=columns_needed, fill_value=0)

# Tahmin yapma
prediction = loaded_model.predict(user_input)

# Tahmin sonucunu gösterme
st.subheader('Tahmin Sonucu')
st.write('İşe Alınma Durumu: {}'.format('Alınacak' if prediction[0] == 1 else 'Alınmayacak'))
