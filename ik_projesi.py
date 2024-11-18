import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Çoklu Dil Desteği
LANGUAGES = {
    "Türkçe": {
        "title": "İşe Alınma Tahmin Uygulaması",
        "position_label": "Pozisyon",
        "age_label": "Yaş",
        "education_label": "Eğitim Seviyesi",
        "experience_label": "Deneyim (Yıl)",
        "companies_label": "Çalıştığı Şirket Sayısı",
        "gender_label": "Cinsiyet",
        "interview_label": "Mülakat Skoru",
        "skill_label": "Beceri Skoru",
        "personality_label": "Kişilik Skoru",
        "recommendations": "Öneriler",
        "result_hire": "✅ İŞE ALINABİLİR",
        "result_not_hire": "❌ İŞE ALINAMAZ",
        "detailed_report": "### Detaylı Tahmin Raporu",
        "similar_candidates": "En Yakın İşe Alınmış Çalışanlar",
        "min_experience_warning": "Bu pozisyon için minimum {position} deneyim gerekliliği karşılanmamaktadır!",
        "image_caption": "Bu uygulama, işe alım sürecinizi desteklemek için geliştirilmiştir.",
    },
    "English": {
        "title": "Hiring Prediction Application",
        "position_label": "Position",
        "age_label": "Age",
        "education_label": "Education Level",
        "experience_label": "Experience (Years)",
        "companies_label": "Number of Companies Worked",
        "gender_label": "Gender",
        "interview_label": "Interview Score",
        "skill_label": "Skill Score",
        "personality_label": "Personality Score",
        "recommendations": "Recommendations",
        "result_hire": "✅ HIREABLE",
        "result_not_hire": "❌ NOT HIREABLE",
        "detailed_report": "### Detailed Prediction Report",
        "similar_candidates": "Most Similar Hired Candidates",
        "min_experience_warning": "The minimum experience requirement for the {position} position is not met!",
        "image_caption": "This application is designed to support your recruitment process.",
    }
}

# Kullanıcı Dilini Belirleme
def get_language():
    return st.sidebar.radio("Dil / Language", options=["Türkçe", "English"])

# Model Eğitme
def train_model():
    data = pd.read_csv('recruitment_data.csv')
    X = data.drop(columns=['HiringDecision'])
    y = data['HiringDecision']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    st.sidebar.write(f"Model Accuracy: {accuracy:.2f}")

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

# Öneriler Oluşturma
def generate_recommendations(position, experience_years, total_score):
    recommendations = []
    if position == "Müdür" and experience_years < 10:
        recommendations.append("Daha fazla deneyim kazanın.")
    if total_score < 60:
        recommendations.append("Mülakat performansınızı artırmaya odaklanın.")
    if len(recommendations) == 0:
        recommendations.append("Her şey harika görünüyor!")
    return recommendations

# En Yakın Çalışanları Bulma
def find_similar_candidates(user_input, data):
    hired_data = data[data['HiringDecision'] == 1].drop(columns=['HiringDecision'])
    user_input = user_input.reindex(columns=hired_data.columns, fill_value=0)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(hired_data)
    user_scaled = scaler.transform(user_input)

    similarity_scores = cosine_similarity(data_scaled, user_scaled).flatten()
    top_indices = similarity_scores.argsort()[-3:][::-1]
    return hired_data.iloc[top_indices]

# Ana Uygulama
def main_app():
    lang = get_language()
    labels = LANGUAGES[lang]

    st.title(labels["title"])

    # Veri Setini Yükle
    data = pd.read_csv('recruitment_data.csv')

    # Model Yükleme veya Eğitme
    if 'model' not in st.session_state:
        st.session_state['model'] = train_model()

    model = st.session_state['model']

    # Kullanıcıdan Veri Alma
    def get_user_input():
        position = st.sidebar.selectbox(labels["position_label"], ['Uzman Yardımcısı', 'Uzman', 'Müdür', 'Direktör', 'Genel Müdür'])
        age = st.sidebar.number_input(labels["age_label"], min_value=18, max_value=65, value=30)
        education = st.sidebar.selectbox(labels["education_label"], ['Önlisans', 'Lisans', 'Yüksek Lisans', 'Doktora'])
        experience = st.sidebar.slider(labels["experience_label"], 0, 40, 5)
        companies_worked = st.sidebar.number_input(labels["companies_label"], min_value=0, max_value=20, value=1)
        gender = st.sidebar.selectbox(labels["gender_label"], ['Erkek', 'Kadın'])
        interview_score = st.sidebar.slider(labels["interview_label"], 0, 100, 50)
        skill_score = st.sidebar.slider(labels["skill_label"], 0, 100, 50)
        personality_score = st.sidebar.slider(labels["personality_label"], 0, 100, 50)

        total_score = (interview_score + skill_score + personality_score) / 3

        education_mapping = {'Önlisans': 1, 'Lisans': 2, 'Yüksek Lisans': 3, 'Doktora': 4}
        gender_mapping = {'Erkek': 0, 'Kadın': 1}

        user_data = {
            'Age': age,
            'Gender': gender_mapping[gender],
            'EducationLevel': education_mapping[education],
            'ExperienceYears': experience,
            'PreviousCompanies': companies_worked,
            'TotalScore': total_score,
        }
        return pd.DataFrame(user_data, index=[0]), position, total_score

    user_input, position, total_score = get_user_input()

    # Pozisyona Göre Gereksinimleri Kontrol Et
    if position == "Müdür" and user_input['ExperienceYears'][0] < 10:
        st.warning(labels["min_experience_warning"].format(position=position))
        return

    # Model Tahmini
    prediction_proba = model.predict_proba(user_input)[0]
    prediction = model.predict(user_input)

    # Tahmin Sonucu
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    if prediction[0] == 1:
        st.success(labels["result_hire"])
    else:
        st.error(labels["result_not_hire"])
    st.markdown("</div>", unsafe_allow_html=True)

    # Öneriler
    st.subheader(labels["recommendations"])
    recommendations = generate_recommendations(position, user_input['ExperienceYears'][0], total_score)
    for rec in recommendations:
        st.write(f"- {rec}")

    # En Yakın Çalışanlar
    similar_candidates = find_similar_candidates(user_input, data)
    st.sidebar.subheader(labels["similar_candidates"])
    for index, candidate in similar_candidates.iterrows():
        st.sidebar.write(f"Yaş: {candidate['Age']}, Deneyim: {candidate['ExperienceYears']} yıl, Şirket Sayısı: {candidate['PreviousCompanies']}")

    # Görsel ve Metin
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.image("https://www.cottgroup.com/images/Zoo/gorsel/insan-kaynaklari-analitigi-ic-gorsel-2.webp", width=400)
    st.markdown(labels["image_caption"])
    st.markdown("</div>", unsafe_allow_html=True)

# Ana Uygulamayı Çalıştır
main_app()
