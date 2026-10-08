"""
Modul Pra-pemrosesan Teks (Preprocessing Pipeline)
====================================================
Tahap 2 dari arsitektur: membersihkan teks mentah dari media sosial
sebelum masuk ke model NLP.

Setiap fungsi dibuat terpisah dan dapat diaudit satu per satu (tidak ada
transformasi tersembunyi) — ini bagian dari upaya menghindari bias:
pengguna bisa melihat persis apa yang diubah pada tiap tahap lewat
fungsi `preprocess_pipeline(..., verbose=True)`.
"""

import re

# ======================================================================
# 1. PEMBERSIHAN REGEX
# ======================================================================
def bersihkan_regex(teks: str) -> str:
    """Hapus mention, URL, hashtag simbol (kata di belakangnya tetap
    disimpan karena bisa memuat konteks), emoji berulang, dan karakter
    non-alfanumerik berlebih."""
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)          # URL
    teks = re.sub(r"@\w+", " ", teks)                       # mention
    teks = re.sub(r"#(\w+)", r"\1", teks)                    # simbol # dibuang, kata disimpan
    teks = re.sub(r"(.)\1{2,}", r"\1\1", teks)              # karakter berulang >2 -> jadi 2 ("baguuuus" -> "baguus")
    teks = re.sub(r"[^\w\s,.!?']", " ", teks)                # simbol aneh/emoji dibuang
    teks = re.sub(r"\s+", " ", teks).strip()                 # spasi ganda
    return teks


# ======================================================================
# 2. NORMALISASI SLANG / BAHASA GAUL -> BAKU
# ======================================================================
KAMUS_SLANG_ID = {
    "bgt": "banget", "bgt.": "banget", "yg": "yang", "dgn": "dengan",
    "tdk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak",
    "krn": "karena", "karna": "karena", "utk": "untuk", "jd": "jadi",
    "sm": "sama", "dr": "dari", "blm": "belum", "udh": "sudah", "udah": "sudah",
    "dah": "sudah", "sdh": "sudah", "aja": "saja", "jg": "juga",
    "tp": "tapi", "tapi2": "tapi", "trs": "terus", "gmn": "bagaimana",
    "gimana": "bagaimana", "knp": "kenapa", "org": "orang", "bbrp": "beberapa",
    "mantul": "mantap betul", "mantab": "mantap", "josss": "bagus",
    "ok": "oke", "oke2": "oke", "bgs": "bagus", "jelek2": "jelek",
    "pgn": "ingin", "pengen": "ingin", "klo": "kalau", "kalo": "kalau",
    "sy": "saya", "gw": "saya", "gue": "saya", "lu": "kamu", "loe": "kamu",
    "cepet": "cepat", "lambt": "lambat", "parah2": "parah",
    "respon": "respons", "cs": "layanan pelanggan",
}

SLANG_EN = {
    "u": "you", "ur": "your", "r": "are", "bc": "because", "cuz": "because",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "luv": "love", "pls": "please", "plz": "please", "thx": "thanks",
    "tho": "though", "btw": "by the way", "imo": "in my opinion",
    "rly": "really", "gr8": "great", "w8": "wait", "bday": "birthday",
}


def normalisasi_slang(teks: str, bahasa: str = "id") -> str:
    kamus = KAMUS_SLANG_ID if bahasa == "id" else SLANG_EN
    kata = teks.split()
    hasil = [kamus.get(k.lower(), k) for k in kata]
    return " ".join(hasil)


# ======================================================================
# 3. TOKENISASI, STOPWORD REMOVAL, STEMMING RINGAN
# ======================================================================
STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan",
    "pada", "adalah", "akan", "atau", "juga", "ada", "saya", "kamu",
    "kita", "mereka", "dia", "nya", "ya", "sih", "kok", "deh", "lah",
    "karena", "jadi", "saja", "tidak", "bukan", "lagi", "sudah", "belum",
    "sangat", "lebih", "paling", "nya", "pun", "oleh", "dalam", "tersebut",
}

STOPWORDS_EN = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "of",
    "in", "on", "at", "for", "with", "this", "that", "it", "i", "you",
    "we", "they", "he", "she", "be", "been", "has", "have", "had", "as",
    "but", "so", "very", "really", "just", "not",
}


def tokenisasi(teks: str) -> list:
    return re.findall(r"[a-zA-Z]+", teks.lower())


def hapus_stopword(token: list, bahasa: str = "id") -> list:
    stopwords = STOPWORDS_ID if bahasa == "id" else STOPWORDS_EN
    return [t for t in token if t not in stopwords]


def stem_ringan_id(kata: str) -> str:
    """Stemming sangat sederhana (potong imbuhan umum). Untuk akurasi
    jauh lebih baik, gunakan pustaka Sastrawi (opsional, lihat README):
        pip install Sastrawi
    """
    for akhiran in ("nya", "kan", "lah", "kah", "pun", "an"):
        if kata.endswith(akhiran) and len(kata) > len(akhiran) + 3:
            kata = kata[: -len(akhiran)]
            break
    for awalan in ("me", "di", "ter", "ber", "pe"):
        if kata.startswith(awalan) and len(kata) > len(awalan) + 3:
            kata = kata[len(awalan):]
            break
    return kata


def _coba_sastrawi():
    """Jika pustaka Sastrawi terpasang, pakai itu (jauh lebih akurat).
    Jika tidak, fallback ke stem_ringan_id di atas."""
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        return StemmerFactory().create_stemmer().stem
    except ImportError:
        return None


_STEMMER_SASTRAWI = _coba_sastrawi()


def stem_token(token: list, bahasa: str = "id") -> list:
    if bahasa != "id":
        return token  # stemming EN tidak diterapkan di versi ini
    if _STEMMER_SASTRAWI:
        return [_STEMMER_SASTRAWI(t) for t in token]
    return [stem_ringan_id(t) for t in token]


# ======================================================================
# 4. PIPELINE LENGKAP (transparan, setiap tahap bisa diaudit)
# ======================================================================
def preprocess_pipeline(teks: str, bahasa: str = "id", verbose: bool = False) -> dict:
    """Mengembalikan dict berisi hasil tiap tahap pipeline, supaya
    pengguna bisa memeriksa transformasi apa saja yang terjadi (anti
    black-box, mengurangi risiko bias tersembunyi)."""
    tahap_1 = bersihkan_regex(teks)
    tahap_2 = normalisasi_slang(tahap_1, bahasa)
    token_mentah = tokenisasi(tahap_2)
    token_bersih = hapus_stopword(token_mentah, bahasa)
    token_stem = stem_token(token_bersih, bahasa)

    hasil = {
        "teks_asli": teks,
        "setelah_regex": tahap_1,
        "setelah_normalisasi_slang": tahap_2,
        "token_mentah": token_mentah,
        "token_tanpa_stopword": token_bersih,
        "token_setelah_stem": token_stem,
        "teks_final": " ".join(token_stem),
    }
    if verbose:
        for k, v in hasil.items():
            print(f"{k:28}: {v}")
    return hasil
