import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# Pozisyona göre minimum deneyim seviyeleri
position_experience_requirements = {
    "Uzman Yardımcısı": 0,
    "Uzman": 2,
    "Müdür": 5,
    "Direktör": 10,
    "Genel Müdür": 15
}

# Modeli eğit ve kaydet
def train_and_save_model(data_path='recruitment_data.csv', model_path='model.pkl'):
    if not os.path.exists(model_path):  # Model zaten yoksa eğit
        data = pd.read_csv(data_path)
        # Gereksiz sütunlar varsa kaldır
        columns_to_drop = ['DistanceFromCompany', 'RecruitmentStrategy']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

        X = data.drop(columns=['HiringDecision'])
        y = data['HiringDecision']

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        st.sidebar.write("Model başarıyla eğitildi ve kaydedildi.")
    else:
        st.sidebar.write("Model daha önce kaydedildi. Mevcut modeli kullanacağım.")

# Modeli yükle
def load_model(model_path='model.pkl'):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        st.error("Model dosyası bulunamadı! Lütfen önce modeli eğitin.")
        return None

# Dinamik kullanıcı girişi oluştur
def get_user_input(feature_names):
    # DistanceFromCompany ve RecruitmentStrategy'yi kaldır
    feature_names = [feature for feature in feature_names if feature not in ['DistanceFromCompany', 'RecruitmentStrategy']]

    st.sidebar.markdown("### Pozisyon Seçimi")
    position = st.sidebar.selectbox('Pozisyon', list(position_experience_requirements.keys()))

    st.sidebar.markdown("### Aday Bilgileri")
    input_data = {}
    missing_fields = []

    for feature in feature_names:
        if feature == 'Age':
            input_data[feature] = st.sidebar.number_input('Yaş (18-65)', min_value=18, max_value=65, value=18)
        elif feature == 'ExperienceYears':
            input_data[feature] = st.sidebar.slider('Deneyim Yılı', 0, 40, 0)
        elif feature == 'EducationLevel':
            education = st.sidebar.selectbox('Eğitim Seviyesi', ['Seçiniz', 'Önlisans', 'Lisans', 'Yüksek Lisans', 'Doktora'])
            mapping = {'Seçiniz': 0, 'Önlisans': 1, 'Lisans': 2, 'Yüksek Lisans': 3, 'Doktora': 4}
            input_data[feature] = mapping[education]
            if input_data[feature] == 0:
                missing_fields.append('Eğitim Seviyesi')
        elif feature == 'Gender':
            gender = st.sidebar.selectbox('Cinsiyet', ['Seçiniz', 'Erkek', 'Kadın'])
            mapping = {'Seçiniz': None, 'Erkek': 0, 'Kadın': 1}
            input_data[feature] = mapping[gender]
            if input_data[feature] is None:
                missing_fields.append('Cinsiyet')
        elif feature == 'InterviewScore':
            input_data[feature] = st.sidebar.slider('Mülakat Skoru (0-100)', 0, 100, 0)
        elif feature == 'SkillScore':
            input_data[feature] = st.sidebar.slider('Beceri Skoru (0-100)', 0, 100, 0)
        elif feature == 'PersonalityScore':
            input_data[feature] = st.sidebar.slider('Kişilik Skoru (0-100)', 0, 100, 0)
        else:
            input_data[feature] = st.sidebar.number_input(feature, min_value=0, max_value=100, value=0)

    return pd.DataFrame(input_data, index=[0]), position, missing_fields

# Ana uygulama
def main_app():
    st.title("İşe Alım Tahmin Uygulaması")
    st.sidebar.markdown("## Model Ayarları")

    st.image("https://www.cottgroup.com/images/Zoo/gorsel/insan-kaynaklari-analitigi-ic-gorsel-2.webp", width=400)

    data_path = 'recruitment_data.csv'
    model_path = 'model.pkl'
    scaler = MinMaxScaler()

    # Modeli eğit veya yükle
    train_and_save_model(data_path, model_path)
    model = load_model(model_path)

    if model is None:
        return

    data = pd.read_csv(data_path)
    # Gereksiz sütunlar varsa kaldır
    columns_to_drop = ['DistanceFromCompany', 'RecruitmentStrategy']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    # Kullanıcıdan veri al
    user_input, position, missing_fields = get_user_input(model.feature_names_in_)

    # Eksik alanlar için uyarılar
    if missing_fields:
        st.warning(f"Lütfen eksik alanları doldurun: {', '.join(missing_fields)}")
        return

    # Pozisyona göre minimum deneyim kontrolü
    required_experience = position_experience_requirements[position]
    if user_input['ExperienceYears'].iloc[0] < required_experience:
        st.warning(f"{position} pozisyonu için minimum {required_experience} yıl deneyim gereklidir.")
        return

    # Tahmin yap
    prediction_proba = model.predict_proba(user_input)[0]
    prediction = model.predict(user_input)

    # Tahmin sonuçlarını göster
    if prediction[0] == 1:
        st.success("✅ Aday İŞE ALINABİLİR")
    else:
        st.error("❌ Aday İŞE ALINAMAZ")

    st.markdown("### Detaylı Değerlendirme")
    st.write(f"İşe alınma olasılığı: %{prediction_proba[1] * 100:.2f}")
    st.write(f"Reddedilme olasılığı: %{prediction_proba[0] * 100:.2f}")

# Uygulamayı çalıştır
if __name__ == "__main__":
    main_app()
