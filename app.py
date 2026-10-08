import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
 
from analisis_sentimen import (
    df, df_en,
    hitung_polaritas, tentukan_label, THRESHOLD,
    analisis_textblob, TEXTBLOB_THRESHOLD,
)
 
st.set_page_config(page_title="Analisis Sentimen", page_icon="💬", layout="centered")
 
EMOJI_ID = {"Positif": "😊", "Netral": "😐", "Negatif": "😠"}
EMOJI_EN = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}
 
 
@st.cache_resource
def muat_model_id():
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000),
    )
    model.fit(df["teks"], df["label"])
    return model
 
 
# ------------------------- Tampilan -------------------------
st.title("💬 Analisis Sentimen Teks")
 
bahasa = st.radio("Pilih bahasa / Choose language:", ["Bahasa Indonesia", "English"], horizontal=True)
 
st.divider()
 
# =====================================================================
# MODE BAHASA INDONESIA
# =====================================================================
if bahasa == "Bahasa Indonesia":
    st.write("Masukkan ulasan produk, film, atau cuitan media sosial berbahasa Indonesia.")
 
    contoh = st.selectbox(
        "Coba contoh kalimat (opsional):",
        ["", "Pelayanan ini sangat memuaskan dan cepat!",
         "Barangnya jelek banget, tidak sesuai foto. Kecewa!",
         "Paket diterima hari Senin pukul sepuluh pagi."],
    )
    teks = st.text_area("Tulis kalimat di sini:", value=contoh, height=120)
 
    with st.sidebar:
        st.header("Pengaturan")
        threshold = st.slider("Threshold polaritas (Kamus)", 0.0, 0.5, float(THRESHOLD), 0.05)
 
    if st.button("Analisis", type="primary"):
        if not teks.strip():
            st.warning("Silakan masukkan kalimat terlebih dahulu.")
        else:
            model = muat_model_id()
            polaritas = hitung_polaritas(teks)
            label_kamus = tentukan_label(polaritas, threshold)
            label_ml = model.predict([teks])[0]
            proba = pd.Series(model.predict_proba([teks])[0], index=model.classes_)
 
            kol1, kol2 = st.columns(2)
            with kol1:
                st.subheader("Metode Kamus")
                st.metric("Polaritas", f"{polaritas:+.2f}")
                st.success(f"{EMOJI_ID[label_kamus]} {label_kamus}")
            with kol2:
                st.subheader("Model Scikit-learn")
                st.metric("Prediksi", label_ml)
                st.success(f"{EMOJI_ID[label_ml]} {label_ml}")
 
            st.caption("Probabilitas model Scikit-learn")
            st.bar_chart(proba)
 
    with st.expander("Lihat dataset Indonesia"):
        st.dataframe(df[["teks", "label"]], use_container_width=True)
        st.caption(f"Total {len(df)} kalimat.")
 
# =====================================================================
# MODE ENGLISH (TextBlob)
# =====================================================================
else:
    st.write("Enter a product review, movie review, or short social media post in English.")
 
    example = st.selectbox(
        "Try an example sentence (optional):",
        ["", "The service is very satisfying and fast!",
         "This product is terrible, nothing like the photos. Disappointed!",
         "The package was delivered on Monday at ten in the morning."],
    )
    text = st.text_area("Write your sentence here:", value=example, height=120)
 
    with st.sidebar:
        st.header("Settings")
        threshold_en = st.slider("Polarity threshold (TextBlob)", 0.0, 0.5, float(TEXTBLOB_THRESHOLD), 0.05)
 
    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Please enter a sentence first.")
        else:
            polarity, subjectivity, label = analisis_textblob(text, threshold_en)
 
            kol1, kol2 = st.columns(2)
            with kol1:
                st.metric("Polarity", f"{polarity:+.2f}")
            with kol2:
                st.metric("Subjectivity", f"{subjectivity:.2f}")
 
            st.success(f"{EMOJI_EN[label]} {label}")
            st.caption(
                "Polarity ranges from -1.0 (very negative) to 1.0 (very positive). "
                "Subjectivity ranges from 0.0 (objective/factual) to 1.0 (subjective/opinion)."
            )
 
    with st.expander("View English dataset"):
        st.dataframe(df_en[["text", "label"]], use_container_width=True)
        st.caption(f"Total {len(df_en)} sentences.")
 