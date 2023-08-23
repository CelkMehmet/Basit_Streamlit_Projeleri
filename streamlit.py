import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



st.set_page_config(page_title="Makine Öğrenimi Uygulaması", page_icon="✅", layout="wide")

# Veri yükleme
data = load_iris()
X = data.data
y = data.target

# Veriyi bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model seçimi
model_names = ["Random Forest", "Gradient Boosting", "SVM"]
selected_model = st.sidebar.selectbox("Bir model seçin:", model_names)

if selected_model == "Random Forest":
    model = RandomForestClassifier()
    params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
elif selected_model == "Gradient Boosting":
    model = GradientBoostingClassifier()
    params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
elif selected_model == "SVM":
    model = SVC()
    params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Hiperparametre ayarlama
tune_params = st.sidebar.checkbox("Hiperparametre ayarlamayı etkinleştir")

if tune_params:
    st.sidebar.write("Hiperparametre ayarlama seçenekleri")
    tuned_param = st.sidebar.selectbox("Bir hiperparametre ayarı seçin:", list(params.keys()))
    selected_value = st.sidebar.selectbox("Seçili hiperparametre değeri:", params[tuned_param])
    params[tuned_param] = [selected_value]  # Parametreleri liste içinde tutmalıyız

    grid_search = GridSearchCV(model, params, cv=3)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

# Modeli eğitme
model.fit(X_train, y_train)

# Tahminler
y_pred = model.predict(X_test)

# Sınıflandırma sonuçları
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

# Sonuçları gösterme
st.title("Makine Öğrenimi Uygulaması")
st.write("Seçilen model:", selected_model)
st.write("Seçilen hiperparametreler:", params)

accuracy = accuracy_score(y_test, y_pred)
st.write("Doğruluk (Accuracy):", accuracy)

# Sınıflandırma sonuçlarını gösterme
st.subheader("Sınıflandırma Sonuçları")
st.text(classification_rep)

# Karmaşıklık Matrisini gösterme
st.subheader("Karmaşıklık Matrisi")
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, cmap="Blues", fmt="d", xticklabels=data.target_names, yticklabels=data.target_names)
st.pyplot(plt) 

# Görselleştirmeler
st.subheader("Veri Örnekleri")
st.write(data.data)

st.subheader("Gerçek Etiketler")
st.write(data.target)

st.subheader("Tahmin Edilen Etiketler")
st.write(y_pred)
