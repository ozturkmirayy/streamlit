import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Modeli eğitme fonksiyonu
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

# Eğer model yoksa bir defaya mahsus oluşturun ve kaydedin
train_model()

# Streamlit sayfa ayarları
st.set_page_config(
    page_title="İşe Alım Tahminleme",
    page_icon="https://www.cottgroup.com/images/Zoo/gorsel/insan-kaynaklari-analitigi-ic-gorsel-2.webp",
)

# Sayfa arka plan rengi ve bileşenlerin stilini özelleştirme
st.markdown("""
    <style>
    .main {
        background-color: #f3f4f6; /* Ana sayfa arka plan rengi */
        color: #333333; /* Genel metin rengi */
    }
    .stSidebar {
        background-color: #e1e4ea; /* Yan menü arka plan rengi */
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50; /* Buton rengi */
    }
    .stNumberInput>label, .stSelectbox>label, .stSlider>label {
        color: #4CAF50; /* Giriş bileşenleri etiket rengi */
    }
    </style>
    """, unsafe_allow_html=True)

# Açıklama ve görseller
st.title("İşe Alınma Tahmin Uygulaması")
st.image("https://www.cottgroup.com/images/Zoo/gorsel/insan-kaynaklari-analitigi-ic-gorsel-2.webp", width=300)
st.markdown("""
    Bu uygulama, adayların işe alınma sürecinde başarıyla değerlendirilip değerlendirilemeyeceğini öngörmek için geliştirilmiştir.
    Modele girilen bilgiler doğrultusunda, bir adayın işe alınma ihtimali "Alınacak" veya "Alınmayacak" olarak belirlenir.
""")
st.markdown("**Önemli Not**: Bu uygulama yalnızca bilgilendirme amaçlıdır. İşe alım kararlarını desteklemek amacıyla kullanılabilir ancak tek başına karar için yeterli değildir.")

# Kullanıcı girişi fonksiyonu (sidebar üzerinden yapılacak girişler)
def get_user_input():
    st.sidebar.header("Aday Bilgileri")
    age = st.sidebar.number_input('Yaş', min_value=18, max_value=65, value=30)
    education = st.sidebar.selectbox('Eğitim Seviyesi', ['Önlisans', 'Lisans', 'Yüksek Lisans', 'Doktora'])
    experience = st.sidebar.slider('Deneyim (Yıl)', 0, 40, 5)
    distance = st.sidebar.slider('Şirketten Uzaklık (km)', 0, 100, 10)
    gender = st.sidebar.selectbox('Cinsiyet', ['Erkek', 'Kadın'])

    # Eğitim seviyesi ve cinsiyet için sayısal değerler
    education_mapping = {'Önlisans': 1, 'Lisans': 2, 'Yüksek Lisans': 3, 'Doktora': 4}
    gender_mapping = {'Erkek': 0, 'Kadın': 1}

    education_num = education_mapping[education]
    gender_num = gender_mapping[gender]

    # Kullanıcı verilerini DataFrame olarak oluşturma
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
prediction = loaded_model.predict(user_input)

# Tahmin sonucunu gösterme
st.subheader('Tahmin Sonucu')
st.write('İşe Alınma Durumu: {}'.format('Alınacak' if prediction[0] == 1 else 'Alınmayacak'))
