import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Tema CSS Kodları
themes = {
    "light": """
        <style>
            body { background-color: #FFFFFF; color: black; }
        </style>
    """,
    "dark": """
        <style>
            body { background-color: #000000; color: white; }
        </style>
    """,
    "colorful": """
        <style>
            body { background-color: #E6E6FA; color: white; }
        </style>
    """
}

# Tema uygulama
def apply_theme(theme):
    st.markdown(themes[theme], unsafe_allow_html=True)

# Veri seti yükleme
def load_data():
    uploaded_file = st.sidebar.file_uploader("Veri Setini Yükleyin (CSV Formatında)", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        st.warning("Lütfen bir veri seti yükleyin.")
        st.stop()

# Ana uygulama
def main_app():
    theme = st.sidebar.selectbox("Tema Seç", ["light", "dark", "colorful"])
    apply_theme(theme)

    st.title("İşe Alınma Tahmin Uygulaması")

    # Veri setini yükle
    data = load_data()

    # Kullanıcıdan veri alma
    def get_user_input():
        age = st.sidebar.number_input('Yaş', min_value=18, max_value=65, value=30)
        education = st.sidebar.selectbox('Eğitim Seviyesi', ['Önlisans', 'Lisans', 'Yüksek Lisans', 'Doktora'])
        experience = st.sidebar.slider('Deneyim (Yıl)', 0, 40, 5)
        previous_companies = st.sidebar.number_input('Çalıştığı Şirket Sayısı', min_value=0, max_value=20, value=1)
        interview_score = st.sidebar.slider('Mülakat Puanı', 0, 100, 50)
        skill_score = st.sidebar.slider('Beceri Puanı', 0, 100, 50)

        education_mapping = {'Önlisans': 1, 'Lisans': 2, 'Yüksek Lisans': 3, 'Doktora': 4}

        user_data = {
            'Age': age,
            'EducationLevel': education_mapping[education],
            'ExperienceYears': experience,
            'PreviousCompanies': previous_companies,
            'InterviewScore': interview_score,
            'SkillScore': skill_score
        }
        return pd.DataFrame(user_data, index=[0])

    user_input = get_user_input()

    # Değerlendirme
    def evaluate_candidate(data):
        if (
            data['ExperienceYears'][0] >= 5 and 
            data['PreviousCompanies'][0] >= 2 and 
            data['InterviewScore'][0] > 50 and 
            data['SkillScore'][0] > 50
        ):
            return "İşe Alınabilir"
        else:
            return "İşe Alınamaz"

    result = evaluate_candidate(user_input)

    # Tahmin sonucu
    if result == "İşe Alınabilir":
        st.success("✅ İŞE ALINABİLİR")
    else:
        st.error("❌ İŞE ALINAMAZ")

# Ana uygulamayı çalıştır
main_app()
