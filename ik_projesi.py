import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
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

# İşe alınamama nedenlerini açıklayan metin oluşturma
def generate_rejection_reason(user_input, required_experience):
    reasons = []

    # Deneyim eksikliği kontrolü
    if user_input.loc[0, 'ExperienceYears'] < required_experience:
        reasons.append(f"Gerekli minimum deneyim yılı: {required_experience}. Adayın deneyimi {user_input.loc[0, 'ExperienceYears']} yıl.")

    # Toplam skor eksikliği kontrolü
    if user_input.loc[0, 'TotalScore'] < 60:
        reasons.append(f"Toplam skor düşük ({user_input.loc[0, 'TotalScore']:.1f}/100).")

    # Eğitim seviyesi eksikliği kontrolü
    if user_input.loc[0, 'EducationLevel'] < 2:
        reasons.append("Eğitim seviyesi, lisans veya daha yüksek düzeyde değil.")

    # Şirket deneyimi kontrolü
    if user_input.loc[0, 'PreviousCompanies'] < 2:
        reasons.append(f"Çalıştığı şirket sayısı düşük ({user_input.loc[0, 'PreviousCompanies']} şirket).")

    # Eğer herhangi bir neden bulunmazsa varsayılan bir metin döndür
    if not reasons:
        reasons.append("Belirgin bir eksiklik bulunmamaktadır ancak genel değerlendirme sonucu olumsuzdur.")

    return reasons

# Pozisyona göre minimum deneyim yılları
position_experience_requirements = {
    "Uzman Yardımcısı": 0,
    "Uzman": 2,
    "Müdür": 5,
    "Direktör": 10,
    "Genel Müdür": 15
}

# Kullanıcıdan veri alma
def get_user_input():
    position = st.sidebar.selectbox('Pozisyon', ['Seçiniz', 'Uzman Yardımcısı', 'Uzman', 'Müdür', 'Direktör', 'Genel Müdür'])
    age = st.sidebar.number_input('Yaş', min_value=18, max_value=65, value=18)
    education = st.sidebar.selectbox('Eğitim Seviyesi', ['Seçiniz', 'Önlisans', 'Lisans', 'Yüksek Lisans', 'Doktora'])
    experience = st.sidebar.slider('Deneyim (Yıl)', 0, 40, 0)
    companies_worked = st.sidebar.number_input('Çalıştığı Şirket Sayısı', min_value=0, max_value=20, value=0)
    gender = st.sidebar.selectbox('Cinsiyet', ['Seçiniz', 'Erkek', 'Kadın'])
    interview_score = st.sidebar.slider('Mülakat Skoru', 0, 100, 0)
    skill_score = st.sidebar.slider('Beceri Skoru', 0, 100, 0)
    personality_score = st.sidebar.slider('Kişilik Skoru', 0, 100, 0)

    # Skorların ortalaması
    total_score = (interview_score + skill_score + personality_score) / 3 if (interview_score + skill_score + personality_score) > 0 else 0

    education_mapping = {'Seçiniz': 0, 'Önlisans': 1, 'Lisans': 2, 'Yüksek Lisans': 3, 'Doktora': 4}
    gender_mapping = {'Seçiniz': 0, 'Erkek': 0, 'Kadın': 1}

    user_data = {
        'Age': age,
        'Gender': gender_mapping[gender],
        'EducationLevel': education_mapping[education],
        'ExperienceYears': experience,
        'PreviousCompanies': companies_worked,
        'TotalScore': total_score
    }

    return pd.DataFrame(user_data, index=[0])

# Ana uygulama
def main_app():
    st.title("İşe Alınma Tahmin Uygulaması")

    # Veri setini yükle
    data = pd.read_csv('recruitment_data.csv')

    # Model yükleme veya eğitme
    if 'model' not in st.session_state:
        st.session_state['model'] = train_model()

    model = st.session_state['model']

    # Kullanıcı verisini al
    user_input = get_user_input()

    # Eksik bilgi kontrolü
    if (user_input[['Age', 'Gender', 'EducationLevel', 'ExperienceYears', 'TotalScore']] == 0).any().any():
        st.info("Lütfen tüm alanları doldurunuz.")
        return

    # Pozisyon seçimi ve minimum deneyim kontrolü
    position = st.sidebar.selectbox('Pozisyon', ['Seçiniz'] + list(position_experience_requirements.keys()))
    if position == 'Seçiniz':
        st.warning("Lütfen pozisyon seçiniz.")
        return

    required_experience = position_experience_requirements[position]
    if user_input.loc[0, 'ExperienceYears'] < required_experience:
        st.warning(f"{position} pozisyonu için minimum {required_experience} yıl deneyim gereklidir.")
        return

    # Tahmin yapma
    prediction_proba = model.predict_proba(user_input)[0]
    prediction = model.predict(user_input)

    # Tahmin sonucunu gösterme
    if prediction[0] == 1:
        st.success("✅ İŞE ALINABİLİR")
    else:
        st.error("❌ İŞE ALINAMAZ")
        st.write("### İşe Alınamama Nedenleri:")
        reasons = generate_rejection_reason(user_input, required_experience)
        for reason in reasons:
            st.write(f"- {reason}")

# Ana uygulamayı çalıştır
main_app()
