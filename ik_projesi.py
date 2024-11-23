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

# En yakın çalışanı bulma
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
        position = st.sidebar.selectbox('Pozisyon', ['Seçiniz', 'Uzman Yardımcısı', 'Uzman', 'Müdür', 'Direktör', 'Genel Müdür'])
        age = st.sidebar.number_input('Yaş', min_value=0, max_value=65, value=0)  # Default 0
        education = st.sidebar.selectbox('Eğitim Seviyesi', ['Seçiniz', 'Önlisans', 'Lisans', 'Yüksek Lisans', 'Doktora'])
        experience = st.sidebar.slider('Deneyim (Yıl)', 0, 40, 0)  # Default 0
        companies_worked = st.sidebar.number_input('Çalıştığı Şirket Sayısı', min_value=0, max_value=20, value=0)  # Default 0
        gender = st.sidebar.selectbox('Cinsiyet', ['Seçiniz', 'Erkek', 'Kadın'])
        interview_score = st.sidebar.slider('Mülakat Skoru', 0, 100, 0)  # Default 0
        skill_score = st.sidebar.slider('Beceri Skoru', 0, 100, 0)  # Default 0
        personality_score = st.sidebar.slider('Kişilik Skoru', 0, 100, 0)  # Default 0

        # Skorların ortalaması
        total_score = (interview_score + skill_score + personality_score) / 3 if interview_score + skill_score + personality_score > 0 else 0

        education_mapping = {'Seçiniz': 0, 'Önlisans': 1, 'Lisans': 2, 'Yüksek Lisans': 3, 'Doktora': 4}
        gender_mapping = {'Seçiniz': 0, 'Erkek': 0, 'Kadın': 1}

        user_data = {
            'Age': age,
            'Gender': gender_mapping[gender],
            'EducationLevel': education_mapping[education],
            'ExperienceYears': experience,
            'PreviousCompanies': companies_worked,
            'DistanceFromCompany': 0,  # Placeholder
            'TotalScore': total_score,
            'RecruitmentStrategy': 1,  # Default value
            'Position': position
        }
        return pd.DataFrame(user_data, index=[0])

    user_input = get_user_input()

    # Pozisyon ve deneyim kontrolü
    if user_input['Position'][0] != 'Seçiniz':
        required_experience = position_experience_requirements[user_input['Position'][0]]
        if user_input['ExperienceYears'][0] < required_experience:
            st.warning(f"{user_input['Position'][0]} pozisyonu için minimum {required_experience} yıl deneyim gereklidir!")

    # Eğer tüm seçimler yapılmadıysa uyarı göster
    if (user_input[['Age', 'Gender', 'EducationLevel', 'ExperienceYears', 'PreviousCompanies', 'TotalScore']] == 0).all(axis=None):
        st.info("Lütfen tüm bilgileri doldurunuz!")
        return

    # Kullanıcı verisini modelin beklediği sütun düzenine göre sıralama
    user_input = user_input.drop(columns=['Position'])  # Pozisyon modelde kullanılmıyor
    user_input = user_input.reindex(columns=model.feature_names_in_, fill_value=0)

    # Tahmin yapma
    prediction_proba = model.predict_proba(user_input)[0]
    prediction = model.predict(user_input)

    # Tahmin sonucunu gösterme
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    if prediction[0] == 1:
        st.success("✅ İŞE ALINABİLİR")
    else:
        st.error("❌ İŞE ALINAMAZ")
    st.markdown("</div>", unsafe_allow_html=True)

    # En yakın çalışanları bulma ve gösterme
    similar_candidates = find_similar_candidates(user_input, data)
    st.sidebar.subheader("En Yakın İşe Alınmış Çalışanlar")
    for index, candidate in similar_candidates.iterrows():
        st.sidebar.write(
            f"Yaş: {int(candidate['Age'])}, Deneyim: {int(candidate['ExperienceYears'])} yıl, Şirket Sayısı: {int(candidate['PreviousCompanies'])}"
        )

    # Tanıtım Metni ve Görsel
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.image("https://www.cottgroup.com/images/Zoo/gorsel/insan-kaynaklari-analitigi-ic-gorsel-2.webp", width=400)
    st.markdown("""
        **Bu uygulama, işe alım sürecinizi desteklemek için geliştirilmiştir.** 
        Adayların deneyimlerini, eğitim seviyelerini ve geçmiş iş bilgilerini kullanarak hızlı bir değerlendirme sağlar.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Ana uygulamayı çalıştır
main_app()
