import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Streamlit ayarları
st.set_page_config(page_title="Hisse Tahmin Uygulaması", page_icon="💹", layout="wide")

# Hisse senedi sembolünü alın
stock_symbol = st.sidebar.text_input("Hisse Senedi Sembolü (örn: AAPL)")

# Seçilebilecek modeller
models = {
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Machine": SVR()
}

selected_model = st.sidebar.selectbox("Bir model seçin:", list(models.keys()))

if stock_symbol:
    # Hisse senedi verilerini çekme
    @st.cache_data  # Önbellekleme, verileri tekrar tekrar çekmemek için
    def get_stock_data(symbol):
        data = yf.download(symbol, period="5y")
        return data

    stock_data = get_stock_data(stock_symbol)

    # Hisse senedi verilerini inceleme
    st.subheader("Hisse Senedi Verileri")
    st.write(stock_data.tail())

    # Veriyi hazırlama ve modelleme
    X = np.array(range(len(stock_data))).reshape(-1, 1)
    y = stock_data["Close"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = models[selected_model]

    # Hiperparametre seçimi
    if selected_model == "Random Forest":
        n_estimators = st.sidebar.slider("n_estimators", 1, 100, value=10)
        model.set_params(n_estimators=n_estimators)
    elif selected_model == "Gradient Boosting":
        learning_rate = st.sidebar.slider("learning_rate", 0.01, 1.0, value=0.1)
        model.set_params(learning_rate=learning_rate)
    elif selected_model == "Support Vector Machine":
        C = st.sidebar.slider("C", 0.1, 10.0, value=1.0)
        model.set_params(C=C)

    model.fit(X_train, y_train)

    # Tahmin yapma
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Tahmin sonuçlarını gösterme
    st.subheader("Tahmin Sonuçları")
    fig, ax = plt.subplots()
    ax.plot(y_test, label="Gerçek Değer")
    ax.plot(y_pred, label="Tahmin Değeri")
    ax.legend()
    st.pyplot(fig)

    st.write("Ortalama Kare Hata (MSE):", mse)
else:
    st.sidebar.write("Lütfen bir hisse senedi sembolü girin.")

