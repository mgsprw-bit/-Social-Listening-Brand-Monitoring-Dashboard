"""
Modul Peningkatan Model (Human-in-the-Loop Learning)
========================================================
PRINSIP UTAMA — kenapa ini TIDAK membiarkan model melatih dirinya
sendiri dari prediksinya sendiri:

Jika model melatih ulang dirinya memakai hasil prediksinya sendiri
sebagai "kebenaran", kesalahan kecil akan terus diperkuat setiap
siklus (feedback loop) sampai menjadi bias besar — model jadi makin
yakin pada pola yang sebetulnya salah. Ini kebalikan dari tujuan
"menghindari bias".

Solusi yang dipakai di sini: model HANYA dilatih ulang dari
(a) dataset awal yang sudah diverifikasi, ditambah
(b) koreksi yang diinput MANUSIA lewat dashboard (tab "Umpan Balik").
Dengan begitu, model tetap bisa membaik seiring waktu, tapi arah
perbaikannya dikendalikan manusia, bukan oleh model itu sendiri.
"""

import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import analisis_sentimen as inti
import database as db

FOLDER_MODEL = "model_tersimpan"
os.makedirs(FOLDER_MODEL, exist_ok=True)

PATH_MODEL = {
    "id": os.path.join(FOLDER_MODEL, "model_sentimen_id.joblib"),
    "en": os.path.join(FOLDER_MODEL, "model_sentimen_en.joblib"),
}


def _dataset_dasar(bahasa: str) -> pd.DataFrame:
    if bahasa == "id":
        return inti.df.rename(columns={"teks": "text"})[["text", "label"]]
    peta = {"Positive": "Positif", "Negative": "Negatif", "Neutral": "Netral"}
    df_en = inti.df_en.copy()
    df_en["label"] = df_en["label"].map(peta)
    return df_en[["text", "label"]]


def susun_data_latih(bahasa: str) -> tuple:
    """Gabungkan dataset dasar (kurasi manual + kombinasi otomatis) dengan
    SELURUH koreksi manusia yang pernah diinput, lalu buang duplikat teks
    (koreksi manusia menang jika ada bentrok dengan dataset dasar, karena
    mencerminkan kebenaran yang sudah divalidasi)."""
    df_dasar = _dataset_dasar(bahasa)
    df_koreksi = db.ambil_koreksi(bahasa)

    if len(df_koreksi) > 0:
        df_koreksi_bersih = df_koreksi.rename(columns={"label_benar": "label"})[["text", "label"]]
        gabungan = pd.concat([df_dasar, df_koreksi_bersih], ignore_index=True)
        gabungan = gabungan.drop_duplicates(subset="text", keep="last")  # koreksi manusia menang
    else:
        gabungan = df_dasar

    return gabungan, len(df_koreksi)


def latih_ulang(bahasa: str) -> dict:
    """Latih ulang model dari dataset dasar + koreksi manusia, simpan ke
    disk dengan joblib (supaya tidak hilang saat server restart), dan
    catat riwayatnya ke database supaya progresnya bisa dipantau."""
    data_latih, jumlah_koreksi = susun_data_latih(bahasa)

    if data_latih["label"].nunique() < 2 or len(data_latih) < 20:
        return {"berhasil": False, "pesan": "Data latih belum cukup (minimal 20 baris, ≥2 label berbeda)."}

    X_train, X_val, y_train, y_val = train_test_split(
        data_latih["text"], data_latih["label"], test_size=0.15, random_state=42,
        stratify=data_latih["label"] if data_latih["label"].value_counts().min() >= 2 else None,
    )

    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    akurasi = accuracy_score(y_val, model.predict(X_val))

    joblib.dump(model, PATH_MODEL[bahasa])
    db.simpan_riwayat_model(bahasa, len(data_latih), jumlah_koreksi, float(akurasi))

    # Tandai koreksi yang baru dipakai supaya riwayat tetap informatif
    df_koreksi = db.ambil_koreksi(bahasa)
    if len(df_koreksi):
        db.tandai_koreksi_terpakai(df_koreksi["id"].tolist())

    return {
        "berhasil": True,
        "akurasi_validasi": round(float(akurasi), 4),
        "jumlah_data_latih": len(data_latih),
        "jumlah_koreksi_manusia": jumlah_koreksi,
    }


def muat_model(bahasa: str):
    """Muat model dari disk jika sudah pernah dilatih; jika belum ada
    sama sekali, latih dulu sekali dari dataset dasar (bootstrap)."""
    path = PATH_MODEL[bahasa]
    if os.path.exists(path):
        return joblib.load(path)
    latih_ulang(bahasa)
    return joblib.load(path)
