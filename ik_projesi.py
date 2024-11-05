import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import os

# Kullanıcı bilgileri
USERNAME = "user"
PASSWORD = "password"

# Giriş durumu kontrolü için oturum durumu ayarlama
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Varsayılan Tema
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

# Tema Uygulama Fonksiyonu
def apply_theme():
    st.markdown(themes[st.session_state['theme']], unsafe_allow_html=True)
    
# Giriş sayfası
def login():
    st.title("Giriş Yap")
    st.write("Lütfen kullanıcı adı ve şifre ile giriş yapın.")
    username = st.text_input("Kullanıcı Adı")
    password = st.text_input("Şifre", type="password")
    
    if st.button("Giriş"):
        if username == USERNAME and password == PASSWORD:
            st.session_state['authenticated'] = True
            st.success("Başarıyla giriş yapıldı!")
        else:
            st.error("Kullanıcı adı veya şifre hatalı.")

# Model eğitme ve kaydetme fonksiyonu
def train_and_save_model():
    data = pd.read_csv('recruitment_data.csv')
    X = data.drop('HiringDecision', axis=1)
    y = data['HiringDecision']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Doğruluğu: {accuracy}")

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Modelin varlığını kontrol et ve yoksa eğitip kaydet
if not os.path.exists('model.pkl'):
    train_and_save_model()

# Ana uygulama sayfası
def main_app():
    # Tema Uygulama
    apply_theme()

    # Tema Butonları
    if st.sidebar.button("Aydınlık Tema"):
        st.session_state['theme'] = 'light'
        apply_theme()
    if st.sidebar.button("Karanlık Tema"):
        st.session_state['theme'] = 'dark'
        apply_theme()
    if st.sidebar.button("Renkli Tema"):
        st.session_state['theme'] = 'colorful'
        apply_theme()

    # Sayfa tasarımı ve stil
    st.markdown("""
        <style>
        .content-card {
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
            text-align: center;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        .result-card {
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
        }
        .result-card.red {
            border-color: #f5c6cb;
        }
        </style>
        """, unsafe_allow_html=True)

    # Başlık, açıklama ve resim kısmı
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>İşe Alınma Tahmin Uygulaması</div>", unsafe_allow_html=True)
    st.write("Adayların iş pozisyonlarına uygunluğunu hızlıca değerlendirin.")
    st.image("https://www.cottgroup.com/images/Zoo/gorsel/insan-kaynaklari-analitigi-ic-gorsel-2.webp", caption="İşe Alım Sürecinde Veriye Dayalı Tahminler")
    st.markdown("</div>", unsafe_allow_html=True)

    # Kullanıcı girişi fonksiyonu
    def get_user_input():
        st.sidebar.header("Aday Bilgileri")
        age = st.sidebar.number_input('Yaş', min_value=18, max_value=65, value=30)
        education = st.sidebar.selectbox('Eğitim Seviyesi', ['Önlisans', 'Lisans', 'Yüksek Lisans', 'Doktora'])
        experience = st.sidebar.slider('Deneyim (Yıl)', 0, 40, 5)
        distance = st.sidebar.slider('Şirketten Uzaklık (km)', 0, 100, 10)
        gender = st.sidebar.selectbox('Cinsiyet', ['Erkek', 'Kadın'])

        education_mapping = {'Önlisans': 1, 'Lisans': 2, 'Yüksek Lisans': 3, 'Doktora': 4}
        gender_mapping = {'Erkek': 0, 'Kadın': 1}

        education_num = education_mapping[education]
        gender_num = gender_mapping[gender]

        user_data = {
            'Age': age, 
            'EducationLevel': education_num, 
            'ExperienceYears': experience,
            'DistanceFromCompany': distance,
            'Gender': gender_num
        }
        
        features = pd.DataFrame(user_data, index=[0])
        return features

    # Kullanıcı girdisini alma
    user_input = get_user_input()

    # Modeli yükleme
    with open('model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # Kullanıcı girdisini modelin ihtiyaç duyduğu sütun düzenine göre yeniden düzenleme
    columns_needed = loaded_model.feature_names_in_
    user_input = user_input.reindex(columns=columns_needed, fill_value=0)

    # Tahmin yapma
    if st.sidebar.button("Tahmin Yap"):
        prediction = loaded_model.predict(user_input)
        display_prediction(prediction, user_input)
        
        # En yakın eşleşen işe alınmış adayın bilgilerini gösterme
        show_closest_match(user_input)

# Tahmin sonucunu ve aday özelliklerini gösterme
def display_prediction(prediction, user_input):
    if prediction[0] == 1:
        result_class = "result-card"
        result_text = "✅ İŞE ALINACAK"
    else:
        result_class = "result-card red"
        result_text = "❌ İŞE ALINMAYACAK"
    
    st.markdown(f"""
        <div class='{result_class}'>
            {result_text}
        </div>
    """, unsafe_allow_html=True)

# En yakın işe alınmış çalışanı gösterme
def show_closest_match(user_input):
    data = pd.read_csv('recruitment_data.csv')
    hired_data = data[data['HiringDecision'] == 1].drop(columns=['HiringDecision'])

    # Tüm özellikleri normalize et
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(hired_data)
    user_scaled = scaler.transform(user_input)

    # En benzer adayı bulma
    similarity_scores = cosine_similarity(data_scaled, user_scaled)
    closest_index = similarity_scores.argmax()
    closest_match = hired_data.iloc[closest_index]

    # Eğitim seviyesini metin olarak getirme
    education_mapping = {1: 'Önlisans', 2: 'Lisans', 3: 'Yüksek Lisans', 4: 'Doktora'}
    closest_education = education_mapping[closest_match['EducationLevel']]

    # En yakın eşleşmeyi görüntüleme
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.subheader("En Yakın Eşleşen Çalışan Bilgileri")
    st.write(f"Yaş: {closest_match['Age']}")
    st.write(f"Eğitim Seviyesi: {closest_education}")
    st.write(f"Deneyim Yılı: {int(closest_match['ExperienceYears'])}")
    st.write(f"Şirketten Uzaklık: {round(closest_match['DistanceFromCompany'])} km")
    st.write(f"Cinsiyet: {'Erkek' if closest_match['Gender'] == 0 else 'Kadın'}")
    st.markdown("</div>", unsafe_allow_html=True)

# Giriş yapılıp yapılmadığını kontrol et
if not st.session_state['authenticated']:
    login()
else:
    main_app()
