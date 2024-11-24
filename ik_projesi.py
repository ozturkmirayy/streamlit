import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# Modeli eğit ve kaydet
def train_and_save_model(data_path='recruitment_data.csv', model_path='model.pkl'):
    if not os.path.exists(model_path):  # Model zaten yoksa eğit
        data = pd.read_csv(data_path)
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

# En yakın işe alınan çalışanları bul
def find_similar_candidates(user_input, data, scaler):
    hired_data = data[data['HiringDecision'] == 1].drop(columns=['HiringDecision'])
    user_input = user_input.reindex(columns=hired_data.columns, fill_value=0)

    # Ölçeklendirme
    data_scaled = scaler.fit_transform(hired_data)
    user_scaled = scaler.transform(user_input)

    # Benzerlik hesapla
    similarity_scores = cosine_similarity(data_scaled, user_scaled).flatten()
    top_indices = similarity_scores.argsort()[-3:][::-1]
    return hired_data.iloc[top_indices]

# Dinamik kullanıcı girişi oluştur
def get_user_input(feature_names):
    st.sidebar.markdown("### Aday Bilgileri")
    input_data = {}
    for feature in feature_names:
        if feature == 'Age':
            input_data[feature] = st.sidebar.number_input('Yaş', min_value=18, max_value=65, value=18)
        elif feature == 'ExperienceYears':
            input_data[feature] = st.sidebar.slider('Deneyim (Yıl)', 0, 40, 0)
        elif feature == 'EducationLevel':
            education = st.sidebar.selectbox('Eğitim Seviyesi', ['Seçiniz', 'Önlisans', 'Lisans', 'Yüksek Lisans', 'Doktora'])
            mapping = {'Seçiniz': 0, 'Önlisans': 1, 'Lisans': 2, 'Yüksek Lisans': 3, 'Doktora': 4}
            input_data[feature] = mapping[education]
        elif feature == 'Gender':
            gender = st.sidebar.selectbox('Cinsiyet', ['Seçiniz', 'Erkek', 'Kadın'])
            mapping = {'Seçiniz': None, 'Erkek': 0, 'Kadın': 1}
            input_data[feature] = mapping[gender]
        elif feature == 'InterviewScore' or feature == 'SkillScore' or feature == 'PersonalityScore':
            input_data[feature] = st.sidebar.slider(f'{feature} (Skor)', 0, 100, 0)
        else:
            input_data[feature] = st.sidebar.number_input(feature, min_value=0, max_value=100, value=0)

    return pd.DataFrame(input_data, index=[0])

# Ana uygulama
def main_app():
    st.title("İşe Alım Tahmin Uygulaması")
    st.sidebar.markdown("## Model Ayarları")

    data_path = 'recruitment_data.csv'
    model_path = 'model.pkl'
    scaler = MinMaxScaler()

    # Modeli eğit veya yükle
    train_and_save_model(data_path, model_path)
    model = load_model(model_path)

    if model is None:
        return

    data = pd.read_csv(data_path)

    # Kullanıcıdan veri al
    user_input = get_user_input(model.feature_names_in_)

    # Eksik bilgi kontrolü
    if user_input.isnull().any(axis=None):
        st.warning("Lütfen tüm alanları eksiksiz doldurun.")
        return

    # Tahmin yap
    prediction_proba = model.predict_proba(user_input)[0]
    prediction = model.predict(user_input)

    # Tahmin sonuçlarını göster
    if prediction[0] == 1:
        st.success("✅ Aday İŞE ALINABİLİR")
        similar_candidates = find_similar_candidates(user_input, data, scaler)
        st.write("Benzer adaylar:")
        st.table(similar_candidates)
    else:
        st.error("❌ Aday İŞE ALINAMAZ")

# Uygulamayı çalıştır
if __name__ == "__main__":
    main_app()
