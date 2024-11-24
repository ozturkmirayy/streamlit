import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Veri seti okuma ve model eğitme
def train_model():
    data = pd.read_csv('recruitment_data.csv')
    X = data.drop(columns=['HiringDecision'])
    y = data['HiringDecision']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    st.sidebar.write(f"Model Doğruluğu: {accuracy:.2f}")

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

# En yakın işe alınan çalışanları bulma
def find_similar_candidates(user_input, data):
    hired_data = data[data['HiringDecision'] == 1].drop(columns=['HiringDecision'])

    # Kullanıcı girişini veri setinin sütunlarıyla eşleştir
    user_input = user_input.reindex(columns=hired_data.columns, fill_value=0)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(hired_data)
    user_scaled = scaler.transform(user_input)

    similarity_scores = cosine_similarity(data_scaled, user_scaled).flatten()
    top_indices = similarity_scores.argsort()[-3:][::-1]
    return hired_data.iloc[top_indices]

# Pozisyona göre minimum deneyim yılları
position_experience_requirements = {
    "Uzman Yardımcısı": 0,
    "Uzman": 2,
    "Müdür": 5,
    "Direktör": 10,
    "Genel Müdür": 15
}

# Ana uygulama
def main_app():
    st.title("İşe Alınma Tahmin Uygulaması")

    # Veri setini yükle
    data = pd.read_csv('recruitment_data.csv')

    # Model yükleme veya eğitme
    if 'model' not in st.session_state:
        st.session_state['model'] = train_model()

    model = st.session_state['model']

    # Kullanıcıdan veri alma
    def get_user_input():
        position = st.sidebar.selectbox('Pozisyon', ['Seçiniz', 'Uzman Yardımcısı', 'Uzman', 'Müdür', 'Direktör', 'Genel Müdür'], key="position_selectbox")
        age = st.sidebar.number_input('Yaş', min_value=18, max_value=65, value=18, key="age_input")
        education = st.sidebar.selectbox('Eğitim Seviyesi', ['Seçiniz', 'Önlisans', 'Lisans', 'Yüksek Lisans', 'Doktora'], key="education_selectbox")
        experience = st.sidebar.slider('Deneyim (Yıl)', 0, 40, 0, key="experience_slider")
        companies_worked = st.sidebar.number_input('Çalıştığı Şirket Sayısı', min_value=0, max_value=20, value=0, key="companies_input")
        gender = st.sidebar.selectbox('Cinsiyet', ['Seçiniz', 'Erkek', 'Kadın'], key="gender_selectbox")
        interview_score = st.sidebar.slider('Mülakat Skoru', 0, 100, 0, key="interview_score_slider")
        skill_score = st.sidebar.slider('Beceri Skoru', 0, 100, 0, key="skill_score_slider")
        personality_score = st.sidebar.slider('Kişilik Skoru', 0, 100, 0, key="personality_score_slider")

        education_mapping = {'Seçiniz': 0, 'Önlisans': 1, 'Lisans': 2, 'Yüksek Lisans': 3, 'Doktora': 4}
        gender_mapping = {'Seçiniz': None, 'Erkek': 0, 'Kadın': 1}

        user_data = {
            'Age': age,
            'Gender': gender_mapping[gender],
            'EducationLevel': education_mapping[education],
            'ExperienceYears': experience,
            'PreviousCompanies': companies_worked
        }
        return pd.DataFrame(user_data, index=[0]), position

    user_input, position = get_user_input()

    # Eksik bilgi kontrolü
    if (
        position == 'Seçiniz'
        or user_input['Gender'].iloc[0] is None
        or user_input['EducationLevel'].iloc[0] == 0
    ):
        st.info("Lütfen tüm alanları doldurunuz. Tahmin yapmak için eksik bilgi olmamalıdır.")
        return

    # Pozisyon için minimum deneyim kontrolü
    required_experience = position_experience_requirements[position]
    if user_input['ExperienceYears'].iloc[0] < required_experience:
        st.warning(f"{position} pozisyonu için minimum {required_experience} yıl deneyim gereklidir.")
        return

    # Kullanıcı verisini modelin beklediği sütun düzenine göre sıralama
    user_input = user_input.reindex(columns=model.feature_names_in_, fill_value=0)

    # Tahmin yapma
    prediction_proba = model.predict_proba(user_input)[0]
    prediction = model.predict(user_input)

    # Tahmin sonucunu gösterme
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    if prediction[0] == 1:
        st.success("✅ İŞE ALINABİLİR")
        similar_candidates = find_similar_candidates(user_input, data)

        # Sütunlar mevcut değilse hata oluşmasını önlemek için kontrol
        for index, candidate in similar_candidates.iterrows():
            st.write(f"- Yaş: {candidate.get('Age', 'Bilinmiyor')}, Deneyim: {candidate.get('ExperienceYears', 'Bilinmiyor')} yıl:.1f}")
    else:
        st.error("❌ İŞE ALINAMAZ")
    st.markdown("</div>", unsafe_allow_html=True)

    # Sağ taraftaki görsel ve yazılar
    st.sidebar.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.image("https://www.cottgroup.com/images/Zoo/gorsel/insan-kaynaklari-analitigi-ic-gorsel-2.webp", width=400)
    st.markdown("""
        **Bu uygulama, işe alım sürecinizi desteklemek için geliştirilmiştir.** 
        Adayların deneyimlerini, eğitim seviyelerini ve geçmiş iş bilgilerini kullanarak hızlı bir değerlendirme sağlar.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Ana uygulamayı çalıştır
main_app()
