import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Kullanıcı bilgileri
USERNAME = "user"
PASSWORD = "password"

# Giriş durumu kontrolü için oturum durumu ayarlama
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

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

# Model eğitimi ve uygulama ana sayfası
def main_app():
    def train_model():
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

    if 'model_trained' not in st.session_state:
        train_model()
        st.session_state['model_trained'] = True

    # Sayfa tasarımı ve stil
    st.markdown("""
        <style>
        .main {
            background-color: #f4f7fb;
            padding: 20px;
        }
        .content-card {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
        }
        .title {
            color: #333333;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        .result-card {
            background-color: #f1f9f1;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #b2d8b2;
            text-align: center;
            color: #155724;
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
        }
        .result-card.red {
            background-color: #fdecea;
            border-color: #f5c6cb;
            color: #721c24;
        }
        </style>
        """, unsafe_allow_html=True)

    # Başlık ve açıklama kartı
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>İşe Alınma Tahmin Uygulaması</div>", unsafe_allow_html=True)
    st.write("Bu uygulama, adayların işe alım sürecinde başarıyla değerlendirilip değerlendirilemeyeceğini öngörmek için geliştirilmiştir.")
    st.write("Lütfen aday bilgilerini girin ve işe alınma tahminini görmek için tahmin butonuna basın.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Kullanıcı girişi fonksiyonu (sidebar üzerinden yapılacak girişler)
    def get_user_input():
        st.sidebar.header("Aday Bilgileri")
        age = st.sidebar.number_input('Yaş', min_value=18, max_value=65, value=30)
        st.sidebar.caption("Adayın yaşını girin (18-65)")
        
        education = st.sidebar.selectbox('Eğitim Seviyesi', ['Önlisans', 'Lisans', 'Yüksek Lisans', 'Doktora'])
        st.sidebar.caption("En yüksek eğitim seviyesini seçin")
        
        experience = st.sidebar.slider('Deneyim (Yıl)', 0, 40, 5)
        st.sidebar.caption("Adayın iş deneyimini yıllık olarak girin")
        
        distance = st.sidebar.slider('Şirketten Uzaklık (km)', 0, 100, 10)
        st.sidebar.caption("Adayın ikamet adresi ile şirket arasındaki mesafeyi girin (km)")
        
        gender = st.sidebar.selectbox('Cinsiyet', ['Erkek', 'Kadın'])
        st.sidebar.caption("Cinsiyet seçimini yapın")

        # Eğitim seviyesi ve cinsiyet için sayısal değerler
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

# Tahmin sonucunu ve aday özelliklerini gösterme
def display_prediction(prediction, user_input):
    if prediction[0] == 1:
        result_class = "result-card"
        result_text = "✅ İŞE ALINABİLİR"
        result_color = "#D4EDDA"
    else:
        result_class = "result-card red"
        result_text = "❌ İŞE ALINMAYABİLİR"
        result_color = "#F8D7DA"
    
    st.markdown(f"""
        <div class='{result_class}'>
            {result_text}
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.subheader("Aday Özellikleri")
    st.write(f"Yaş: {user_input['Age'][0]}")
    st.write(f"Eğitim Seviyesi: {user_input['EducationLevel'][0]}")
    st.write(f"Deneyim Yılı: {user_input['ExperienceYears'][0]}")
    st.write(f"Şirketten Uzaklık: {user_input['DistanceFromCompany'][0]} km")
    st.write(f"Cinsiyet: {'Erkek' if user_input['Gender'][0] == 0 else 'Kadın'}")
    st.markdown("</div>", unsafe_allow_html=True)

# Giriş yapılıp yapılmadığını kontrol et
if not st.session_state['authenticated']:
    login()
else:
    main_app()
