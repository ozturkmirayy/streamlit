import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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

# Ana uygulama
def main_app():
    # Tema uygulama
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

        user_data = {
            'Age': age,
            'EducationLevel': education,
            'ExperienceYears': experience,
            'CompaniesWorked': companies_worked,
            'Gender': gender,
        }
        return pd.DataFrame(user_data, index=[0])

    user_input = get_user_input()

    # Basit işe alım kriterleri
    def evaluate_candidate(data):
        # Basit kurallar:
        # Deneyim >= 5 yıl ve Çalıştığı Şirket Sayısı >= 2 ise "İşe Alınabilir"
        # Aksi halde "İşe Alınamaz"
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

# Ana uygulamayı çalıştır
main_app()
