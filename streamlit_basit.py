import streamlit as st

# Başlık
st.title("Merhaba, Streamlit!")

# Metin
st.write("Bu bir Streamlit uygulamasının basit bir örneğidir.")

# Kullanıcı girişi
user_input = st.text_input("Lütfen bir metin girin:")
st.write("Girdiğiniz metin:", user_input)

# Buton
if st.button("Merhaba"):
    st.write("Butona tıkladınız!")

# Seçenekler
option = st.selectbox("Bir seçenek seçin:", ["Seçenek 1", "Seçenek 2", "Seçenek 3"])
st.write("Seçilen seçenek:", option)
