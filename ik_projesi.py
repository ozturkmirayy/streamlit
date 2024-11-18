!pip install xgboost
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import os
import json
import matplotlib.pyplot as plt

# Kullanıcı bilgileri için JSON dosyası
USER_DB_FILE = 'users.json'

# Kullanıcı veritabanını yükleme
def load_user_db():
    if not os.path.exists(USER_DB_FILE):
        return {}
    with open(USER_DB_FILE, 'r') as file:
        return json.load(file)

# Kullanıcı doğrulama
def authenticate(username, password):
    users = load_user_db()
    if username in users and users[username] == password:
        return True
    return False

# Yeni kullanıcı ekleme
def add_user(username, password):
    users = load_user_db()
    users[username] = password
    with open(USER_DB_FILE, 'w') as file:
        json.dump(users, file)

# Varsayılan oturum ve tema ayarları
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
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

# Giriş sayfası
def login():
    st.title("Giriş Yap")
    username = st.text_input("Kullanıcı Adı")
    password = st.text_input("Şifre", type="password")
    
    if st.button("Giriş"):
        if authenticate(username, password):
            st.session_state['authenticated'] = True
            st.success("Başarıyla giriş yapıldı!")
        else:
            st.error("Kullanıcı adı veya şifre hatalı.")
    
    if st.button("Kayıt Ol"):
        new_username = st.text_input("Yeni Kullanıcı Adı", key="new_user")
        new_password = st.text_input("Yeni Şifre", type="password", key="new_pass")
        if new_username and new_password:
            add_user(new_username, new_password)
            st.success("Kullanıcı başarıyla eklendi!")

# Model eğitme ve kaydetme fonksiyonu
def train_and_save_model(selected_algorithm):
    data = pd.read_csv('recruitment_data.csv')
    X = data.drop('HiringDecision', axis=1)
    y = data['HiringDecision']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if selected_algorithm == "Random Forest":
        model = RandomForestClassifier()
    elif selected_algorithm == "Logistic Regression":
        model = LogisticRegression()
    elif selected_algorithm == "XGBoost":
        model = XGBClassifier()
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"Model Doğruluğu: {accuracy}")

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Tahmin sonucu görselleştirme
def plot_prediction(prediction):
    labels = ['Alınmayacak', 'Alınacak']
    values = [1 - prediction[0], prediction[0]]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['red', 'green'])
    st.pyplot(fig)

# En yakın çalışan bilgilerini gösterme
def show_closest_match(user_input):
    data = pd.read_csv('recruitment_data.csv')
    hired_data = data[data['HiringDecision'] == 1].drop(columns=['HiringDecision'])

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(hired_data)
    user_scaled = scaler.transform(user_input)

    similarity_scores = cosine_similarity(data_scaled, user_scaled)
    closest_index = similarity_scores.argmax()
    closest_match = hired_data.iloc[closest_index]

    education_mapping = {1: 'Önlisans', 2: 'Lisans', 3: 'Yüksek Lisans', 4: 'Doktora'}
    closest_education = education_mapping[closest_match['EducationLevel']]

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.subheader("En Yakın Eşleşen Çalışan Bilgileri")
    st.write(f"Yaş: {closest_match['Age']}")
    st.write(f"Eğitim Seviyesi: {closest_education}")
    st.write(f"Deneyim Yılı: {int(closest_match['ExperienceYears'])}")
    st.write(f"Şirketten Uzaklık: {round(closest_match['DistanceFromCompany'])} km")
    st.write(f"Cinsiyet: {'Erkek' if closest_match['Gender'] == 0 else 'Kadın'}")
    st.markdown("</div>", unsafe_allow_html=True)

# Ana uygulama
def main_app():
    apply_theme()
    
    algorithm = st.sidebar.selectbox("Model Algoritması", ["Random Forest", "Logistic Regression", "XGBoost"])
    
    if st.sidebar.button("Model Güncelle"):
        train_and_save_model(algorithm)
        st.success("Model başarıyla güncellendi!")

    st.sidebar.button("Aydınlık Tema", on_click=lambda: st.session_state.update({'theme': 'light'}))
    st.sidebar.button("Karanlık Tema", on_click=lambda: st.session_state.update({'theme': 'dark'}))
    st.sidebar.button("Renkli Tema", on_click=lambda: st.session_state.update({'theme': 'colorful'}))

    st.title("İşe Alınma Tahmin Uygulaması")

    # Kullanıcıdan veri alma
    def get_user_input():
        age = st.sidebar.number_input('Yaş', min_value=18, max_value=65, value=30)
        education = st.sidebar.selectbox('Eğitim Seviyesi', ['Önlisans', 'Lisans', 'Yüksek Lisans', 'Doktora'])
        experience = st.sidebar.slider('Deneyim (Yıl)', 0, 40, 5)
        distance = st.sidebar.slider('Şirketten Uzaklık (km)', 0, 100, 10)
        gender = st.sidebar.selectbox('Cinsiyet', ['Erkek', 'Kadın'])

        education_mapping = {'Önlisans': 1, 'Lisans': 2, 'Yüksek Lisans': 3, 'Doktora': 4}
        gender_mapping = {'Erkek': 0, 'Kadın': 1}

        user_data = {
            'Age': age,
            'EducationLevel': education_mapping[education],
            'ExperienceYears': experience,
            'DistanceFromCompany': distance,
            'Gender': gender_mapping[gender],
        }
        return pd.DataFrame(user_data, index=[0])

    user_input = get_user_input()
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    columns_needed = model.feature_names_in_
    user_input = user_input.reindex(columns=columns_needed, fill_value=0)
    prediction = model.predict(user_input)

    if prediction[0] == 1:
        st.success("✅ İŞE ALINACAK")
    else:
        st.error("❌ İŞE ALINMAYACAK")

    plot_prediction(prediction)
    show_closest_match(user_input)

if not st.session_state['authenticated']:
    login()
else:
    main_app()
