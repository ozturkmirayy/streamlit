import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Varsayılan tema ayarları
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'

# Tema CSS Kodları
themes = {
    "light": """
        <style>
            body { background-color: #FFFFFF; color: black; }
            .content-card, .result-card { background-color: white; color: black; }
            .title { color: black; }
        </style>
    """,
    "dark": """
        <style>
            body { background-color: #000000; color: white; }
            .content-card, .result-card { background-color: #333333; color: white; }
            .title { color: white; }
        </style>
    """,
    "colorful": """
        <style>
            body { background-color: #E6E6FA; color: white; }
            .content-card, .result-card { background-color: #9370DB; color: white; }
            .title { color: white; }
        </style>
    """
}

# Tema uygulama
def apply_theme():
    st.markdown(themes[st.session_state['theme']], unsafe_allow_html=True)

# En yakın çalışanı bulma
def find_similar_candidates(user_input):
    data = pd.read_csv('recruitment_data.csv')
    hired_data = data[data['HiringDecision'] == 1].drop(columns=['HiringDecision'])

    # Kullanıcı girişini veri setinin sütunlarıyla eşleştir
    user_input = user_input.reindex(columns=hired_data.columns, fill_value=0)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(hired_data)  # Eğitim verisini ölçekle
    user_scaled = scaler.transform(user_input)     # Kullanıcı girişini ölçekle

    # Benzerlik hesaplama
    similarity_scores = cosine_similarity(data_scaled, user_scaled).flatten()
    top_indices = similarity_scores.argsort()[-3:][::-1]  # En benzer ilk 3
    return hired_data.iloc[top_indices]

# Ana uygulama
def main_app():
    apply_theme()
    
    # Tema butonları
    st.sidebar.button("Aydınlık Tema", on_click=lambda: st.session_state.update({'theme': 'light'}))
    st.sidebar.button("Karanlık Tema", on_click=lambda: st.session_state.update({'theme': 'dark'}))
    st.sidebar.button("Renkli Tema", on_click=lambda: st.session_state.update({'theme': 'colorful'}))

    st.title("İşe Alınma Tahmin Uygulaması")

    # Kullanıcıdan veri alma
    def get_user_input():
        age = st.sidebar.number_input('Yaş', min_value=18, max_value=65, value=30)
        education = st.sidebar.selectbox('Eğitim Seviyesi', ['Önlisans', 'Lisans', 'Yüksek Lisans', 'Doktora'])
        experience = st.sidebar.slider('Deneyim (Yıl)', 0, 40, 5)
        companies_worked = st.sidebar.number_input('Çalıştığı Şirket Sayısı', min_value=0, max_value=20, value=1)
        gender = st.sidebar.selectbox('Cinsiyet', ['Erkek', 'Kadın'])

        education_mapping = {'Önlisans': 1, 'Lisans': 2, 'Yüksek Lisans': 3, 'Doktora': 4}
        gender_mapping = {'Erkek': 0, 'Kadın': 1}

        user_data = {
            'Age': age,
            'EducationLevel': education_mapping[education],
            'ExperienceYears': experience,
            'CompaniesWorked': companies_worked,
            'Gender': gender_mapping[gender],
        }
        return pd.DataFrame(user_data, index=[0])

    user_input = get_user_input()

    # Basit işe alım kriterleri
    def evaluate_candidate(data):
        if data['ExperienceYears'][0] >= 5 and data['CompaniesWorked'][0] >= 2:
            return "İşe Alınabilir"
        else:
            return "İşe Alınamaz"

    result = evaluate_candidate(user_input)

    # Tahmin sonucunu gösterme
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    if result == "İşe Alınabilir":
        st.success("✅ İŞE ALINABİLİR")
    else:
        st.error("❌ İŞE ALINAMAZ")
    st.markdown("</div>", unsafe_allow_html=True)

    # En yakın çalışanları bulma ve gösterme
    similar_candidates = find_similar_candidates(user_input)
    st.sidebar.subheader("En Yakın İşe Alınmış Çalışanlar")
    for index, candidate in similar_candidates.iterrows():
        st.sidebar.write(f"Yaş: {candidate['Age']}, Deneyim: {candidate['ExperienceYears']} yıl, Şirket Sayısı: {candidate['CompaniesWorked']}")

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
