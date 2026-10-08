"""
Modul Ekstraksi Metrik Kuantitatif (NSS, Emosi, ABSA, Topic Modeling)
=======================================================================
CATATAN METODOLOGI (penting untuk menghindari bias analisis):
- Deteksi emosi & ABSA di sini berbasis KAMUS KATA KUNCI (lexicon-based),
  BUKAN model machine learning terlatih atas data beremosi berlabel.
  Artinya: akurasinya terbatas pada kata yang ada di kamus, dan TIDAK
  bisa menangkap sarkasme, konteks tersirat, atau kata yang tidak terdaftar.
- Setiap fungsi di bawah akan mengembalikan "Tidak terdeteksi" / confidence
  rendah ketika tidak ada kata kunci yang cocok, alih-alih memaksakan
  sebuah label — ini mencegah sistem "berpura-pura yakin" padahal menebak.
- Untuk analisis yang dipakai sebagai dasar keputusan penting (krisis
  komunikasi, kebijakan publik), hasil otomatis ini SEBAIKNYA disilangkan
  dengan tinjauan manusia atas sampel acak, bukan dipakai mentah-mentah.
"""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# ======================================================================
# 1. NET SENTIMENT SCORE (NSS)
# ======================================================================
def hitung_nss(label_series: pd.Series, label_positif="Positif", label_negatif="Negatif") -> float:
    """NSS = (Total Positif - Total Negatif) / Total Dokumen * 100
    Rentang: -100 (seluruhnya negatif) sampai +100 (seluruhnya positif)."""
    total = len(label_series)
    if total == 0:
        return 0.0
    pos = (label_series == label_positif).sum()
    neg = (label_series == label_negatif).sum()
    return round((pos - neg) / total * 100, 2)


# ======================================================================
# 2. DETEKSI EMOSI (lexicon-based, multi-bahasa)
# ======================================================================
EMOTION_LEXICON_ID = {
    "Senang (Joy)": {"senang", "bahagia", "gembira", "suka", "puas", "seru",
                      "mantap", "bangga", "lega", "terharu", "menyenangkan"},
    "Marah (Anger)": {"marah", "kesal", "jengkel", "geram", "benci", "emosi",
                       "murka", "sebal", "dongkol"},
    "Takut (Fear)": {"takut", "khawatir", "cemas", "waswas", "ngeri", "panik"},
    "Sedih (Sadness)": {"sedih", "kecewa", "menyesal", "putus", "asa", "terpuruk",
                         "menangis", "galau", "murung"},
    "Terkejut (Surprise)": {"kaget", "terkejut", "mengejutkan", "tak", "disangka",
                             "wow", "mencengangkan"},
    "Jijik (Disgust)": {"jijik", "menjijikkan", "muak", "eneg"},
}

EMOTION_LEXICON_EN = {
    "Joy": {"happy", "glad", "joyful", "satisfied", "great", "pleased",
             "delighted", "excited", "proud", "love", "enjoy"},
    "Anger": {"angry", "furious", "annoyed", "irritated", "mad", "hate",
               "frustrated", "outraged"},
    "Fear": {"afraid", "worried", "anxious", "scared", "nervous", "panic"},
    "Sadness": {"sad", "disappointed", "regret", "upset", "depressed",
                 "heartbroken", "unhappy"},
    "Surprise": {"surprised", "shocked", "astonished", "unexpected", "wow"},
    "Disgust": {"disgusted", "gross", "revolting", "nasty"},
}


def deteksi_emosi(teks: str, bahasa: str = "id") -> str:
    """Mengembalikan label emosi dengan jumlah kata kunci terbanyak.
    Jika tidak ada kata kunci yang cocok sama sekali, kembalikan
    'Tidak terdeteksi' (bukan menebak paksa salah satu emosi)."""
    leksikon = EMOTION_LEXICON_ID if bahasa == "id" else EMOTION_LEXICON_EN
    kata = set(teks.lower().split())

    skor = {emosi: len(kata & daftar_kata) for emosi, daftar_kata in leksikon.items()}
    skor_maks = max(skor.values()) if skor else 0

    if skor_maks == 0:
        return "Tidak terdeteksi"

    # Jika ada lebih dari satu emosi dengan skor tertinggi yang sama,
    # jangan asal pilih satu secara acak -> laporkan sebagai "Campuran"
    # supaya tidak menyesatkan pengguna dengan kepastian palsu.
    emosi_tertinggi = [e for e, s in skor.items() if s == skor_maks]
    if len(emosi_tertinggi) > 1:
        return "Campuran (" + " / ".join(emosi_tertinggi) + ")"
    return emosi_tertinggi[0]


# ======================================================================
# 3. ASPECT-BASED SENTIMENT ANALYSIS (ABSA) SEDERHANA
# ======================================================================
ASPEK_KEYWORDS_ID = {
    "Harga": {"harga", "mahal", "murah", "terjangkau", "diskon", "promo"},
    "Pelayanan": {"pelayanan", "layanan", "staf", "karyawan", "respons", "ramah", "cs"},
    "Kualitas": {"kualitas", "bahan", "mutu", "awet", "tahan"},
    "Pengiriman": {"pengiriman", "kirim", "paket", "kurir", "ongkir", "ekspedisi"},
    "Rasa/Produk": {"rasa", "enak", "produk", "barang", "fitur", "desain"},
}

ASPEK_KEYWORDS_EN = {
    "Price": {"price", "expensive", "cheap", "affordable", "discount", "promo"},
    "Service": {"service", "staff", "support", "response", "friendly", "customer"},
    "Quality": {"quality", "material", "durable", "built"},
    "Delivery": {"delivery", "shipping", "courier", "package", "shipment"},
    "Product/Taste": {"taste", "product", "item", "feature", "design"},
}


def pisah_klausa(teks: str) -> list:
    """Pisahkan kalimat jadi klausa pendek berdasarkan tanda baca/konjungsi,
    supaya polaritas per aspek dihitung dari konteks terdekat kata aspek,
    bukan dari keseluruhan kalimat yang bisa mencampur beberapa topik."""
    import re
    potongan = re.split(r"[.,;!?]|(?:\bdan\b)|(?:\btapi\b)|(?:\bnamun\b)|(?:\bbut\b)|(?:\band\b)", teks.lower())
    return [p.strip() for p in potongan if p.strip()]


def ekstrak_sentimen_aspek(daftar_teks: list, fungsi_polaritas, bahasa: str = "id") -> pd.DataFrame:
    """Untuk setiap dokumen, cari klausa yang menyebut kata kunci aspek,
    lalu hitung polaritas klausa tersebut saja (bukan seluruh kalimat).
    `fungsi_polaritas(teks) -> float (-1..1)` disuntik dari luar supaya
    bisa pakai mesin kamus (ID) atau TextBlob (EN) secara seragam."""
    kamus_aspek = ASPEK_KEYWORDS_ID if bahasa == "id" else ASPEK_KEYWORDS_EN
    catatan = {aspek: [] for aspek in kamus_aspek}

    for teks in daftar_teks:
        for klausa in pisah_klausa(teks):
            kata_klausa = set(klausa.split())
            for aspek, kata_kunci in kamus_aspek.items():
                if kata_klausa & kata_kunci:
                    catatan[aspek].append(fungsi_polaritas(klausa))

    baris = []
    for aspek, daftar_skor in catatan.items():
        if daftar_skor:
            baris.append({
                "aspek": aspek,
                "skor_rata_rata": round(float(np.mean(daftar_skor)), 3),
                "jumlah_disebut": len(daftar_skor),
            })
        else:
            baris.append({"aspek": aspek, "skor_rata_rata": None, "jumlah_disebut": 0})

    return pd.DataFrame(baris).sort_values("jumlah_disebut", ascending=False).reset_index(drop=True)


# ======================================================================
# 4. TOPIC MODELING (LDA — Latent Dirichlet Allocation)
# ======================================================================
def pemodelan_topik(daftar_teks: list, bahasa: str = "id", n_topik: int = 5, n_kata: int = 8):
    """Mengelompokkan dokumen ke beberapa topik memakai LDA (scikit-learn).
    Catatan: LDA berbasis bag-of-words, tidak memahami makna semantik
    seperti model Transformer (mis. BERTopic) — topik yang dihasilkan
    adalah kumpulan kata yang SERING MUNCUL BERSAMA, bukan 'tema' yang
    dipahami secara makna oleh mesin."""
    teks_bersih = [t for t in daftar_teks if t and len(t.split()) >= 2]
    if len(teks_bersih) < n_topik * 2:
        return [], "Data terlalu sedikit untuk pemodelan topik yang bermakna (minimal ±{} dokumen).".format(n_topik * 2)

    stopwords = "english" if bahasa != "id" else None
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words=stopwords)
    try:
        matriks = vectorizer.fit_transform(teks_bersih)
    except ValueError:
        return [], "Kosakata terlalu sedikit/seragam untuk membentuk topik."

    lda = LatentDirichletAllocation(n_components=n_topik, random_state=42, max_iter=20)
    lda.fit(matriks)

    kata_fitur = vectorizer.get_feature_names_out()
    topik_hasil = []
    for idx, komponen in enumerate(lda.components_):
        indeks_teratas = komponen.argsort()[::-1][:n_kata]
        kata_teratas = [kata_fitur[i] for i in indeks_teratas]
        topik_hasil.append({"topik": f"Topik {idx + 1}", "kata_kunci": kata_teratas})

    return topik_hasil, None


def kata_terpopuler(daftar_teks: list, bahasa: str = "id", n: int = 30) -> Counter:
    """Hitung frekuensi kata untuk word cloud / daftar isu terpopuler."""
    from preprocessing import tokenisasi, hapus_stopword

    semua_token = []
    for teks in daftar_teks:
        token = hapus_stopword(tokenisasi(teks), bahasa)
        semua_token.extend([t for t in token if len(t) > 2])
    return Counter(semua_token).most_common(n)
