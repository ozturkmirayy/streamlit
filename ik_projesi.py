import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Theme CSS Code
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

# Theme Application
def apply_theme(theme):
    st.markdown(themes[theme], unsafe_allow_html=True)

# Find Similar Candidates
def find_similar_candidates(user_input, data):
    hired_data = data[data['HiringDecision'] == 1].drop(columns=['HiringDecision'])

    # Align user_input with dataset columns
    user_input = user_input.reindex(columns=hired_data.columns, fill_value=0)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(hired_data)
    user_scaled = scaler.transform(user_input)

    similarity_scores = cosine_similarity(data_scaled, user_scaled).flatten()
    top_indices = similarity_scores.argsort()[-3:][::-1]
    return hired_data.iloc[top_indices]

# Main Application
def main_app():
    theme = st.sidebar.selectbox("Tema Seç", ["light", "dark", "colorful"])
    apply_theme(theme)

    st.title("İşe Alınma Tahmin Uygulaması")

    # User Input
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

    # Load Dataset
    data = pd.read_csv('/mnt/data/recruitment_data (1).csv')

    # Evaluation Logic
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

    # Display Result
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    if result == "İşe Alınabilir":
        st.success("✅ İŞE ALINABİLİR")
    else:
        st.error("❌ İŞE ALINAMAZ")
    st.markdown("</div>", unsafe_allow_html=True)

    # Display Similar Candidates
    similar_candidates = find_similar_candidates(user_input, data)
    st.sidebar.subheader("En Yakın İşe Alınmış Çalışanlar")
    for index, candidate in similar_candidates.iterrows():
        st.sidebar.write(f"Yaş: {candidate['Age']}, Deneyim: {candidate['ExperienceYears']} yıl, Şirket Sayısı: {candidate['PreviousCompanies']}")

    # Project Description
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.image("https://www.cottgroup.com/images/Zoo/gorsel/insan-kaynaklari-analitigi-ic-gorsel-2.webp", width=400)
    st.markdown("""
        **Bu uygulama, işe alım sürecinizi desteklemek için geliştirilmiştir.** 
        Adayların deneyimlerini, eğitim seviyelerini ve geçmiş iş bilgilerini kullanarak hızlı bir değerlendirme sağlar.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Run Main Application
main_app()
