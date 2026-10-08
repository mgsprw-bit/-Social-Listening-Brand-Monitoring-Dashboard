"""
Analisis Sentimen Teks — Bahasa Indonesia & English
- Bagian 1: Dataset ulasan berlabel kecil (ditulis manual) + dataset besar
  hasil kombinasi otomatis (1.000 Positif / 1.000 Negatif / 1.000 Netral,
  untuk masing-masing bahasa Indonesia dan English)
- Bagian 2: Analisis TextBlob (English)
- Bagian 3: Analisis berbasis kamus + threshold polaritas (-1.0 s.d. 1.0) (Indonesia)
- Bagian 4: Model Machine Learning (Scikit-learn: TF-IDF + Logistic Regression)
- Bagian 5: Mode interaktif
 
Instalasi (Colab/Terminal):
    pip install textblob scikit-learn pandas
"""
 
import itertools
import random
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
 
# ======================================================================
# 1. DATASET (teks, label)
# ======================================================================
DATASET = [
    # ---------- POSITIF: produk ----------
    ("Barangnya bagus banget, sesuai dengan foto dan pengiriman cepat!", "Positif"),
    ("Kualitas produk sangat memuaskan, harga juga terjangkau.", "Positif"),
    ("Penjualnya ramah dan responsif, pasti belanja lagi di sini.", "Positif"),
    ("Baterai awet, layar jernih, saya suka banget sama hp ini.", "Positif"),
    ("Packing rapi dan aman, barang sampai dengan selamat. Mantap!", "Positif"),
    ("Sepatunya nyaman dipakai seharian, ukurannya pas.", "Positif"),
    ("Pelayanan ini sangat memuaskan dan cepat!", "Positif"),
    # ---------- POSITIF: film ----------
    ("Filmnya seru banget, alur ceritanya menarik dari awal sampai akhir.", "Positif"),
    ("Akting para pemainnya luar biasa, saya sampai terharu.", "Positif"),
    ("Sinematografinya indah dan musiknya keren, sangat direkomendasikan!", "Positif"),
    ("Film ini menghibur sekali, endingnya tidak terduga dan memuaskan.", "Positif"),
    # ---------- POSITIF: media sosial ----------
    ("Hari ini senang banget, akhirnya lulus sidang skripsi!", "Positif"),
    ("Makanan di warung ini enak parah, porsinya juga banyak.", "Positif"),
    ("Kafe barunya nyaman dan pelayanannya baik, wajib dicoba.", "Positif"),
    ("Terima kasih kang ojek, ramah dan sopan sekali!", "Positif"),
    ("Aplikasinya mudah digunakan dan tampilannya bagus.", "Positif"),
    ("Liburan kemarin menyenangkan, pemandangannya indah banget.", "Positif"),
    ("Dosennya baik dan penjelasannya mudah dipahami.", "Positif"),
    ("Bangga sama timnas, mainnya keren dan penuh semangat!", "Positif"),
    ("Hotelnya bersih, staf ramah, sarapannya enak. Puas sekali.", "Positif"),
 
    # ---------- NEGATIF: produk ----------
    ("Barangnya jelek banget, tidak sesuai dengan foto. Kecewa!", "Negatif"),
    ("Kualitas buruk, baru dipakai seminggu sudah rusak.", "Negatif"),
    ("Pengiriman lambat sekali dan paket datang dalam kondisi penyok.", "Negatif"),
    ("Penjualnya tidak ramah dan lama membalas chat.", "Negatif"),
    ("Baterai boros dan hp sering panas, sangat mengecewakan.", "Negatif"),
    ("Sepatunya tidak nyaman dan ukurannya kekecilan. Menyesal beli.", "Negatif"),
    ("Pelayanan ini sangat buruk dan lambat!", "Negatif"),
    # ---------- NEGATIF: film ----------
    ("Filmnya membosankan banget, ceritanya lambat dan mudah ditebak.", "Negatif"),
    ("Akting pemainnya kaku dan dialognya jelek, buang-buang waktu.", "Negatif"),
    ("Endingnya mengecewakan, sayang sekali padahal awalnya bagus.", "Negatif"),
    ("Film terburuk yang pernah saya tonton, tidak recommended.", "Negatif"),
    # ---------- NEGATIF: media sosial ----------
    ("Kesal banget, macet parah dan telat masuk kerja lagi.", "Negatif"),
    ("Makanannya hambar dan mahal, pelayanannya juga lama.", "Negatif"),
    ("Aplikasinya sering error dan bikin frustrasi.", "Negatif"),
    ("Sinyal jelek banget hari ini, bikin emosi.", "Negatif"),
    ("Hotelnya kotor, kamar bau, stafnya cuek. Sangat kecewa.", "Negatif"),
    ("Dosennya galak dan penjelasannya susah dimengerti.", "Negatif"),
    ("Sedih banget, tiketnya habis dan saya tidak jadi berangkat.", "Negatif"),
    ("Layanan pelanggan tidak membantu sama sekali, benar-benar buruk.", "Negatif"),
    ("Parkirnya semrawut dan petugasnya tidak sopan.", "Negatif"),
    ("kamu tidak pantas dan bodoh.", "Negatif"),
 
    # ---------- NETRAL ----------
    ("Paket diterima hari Senin pukul sepuluh pagi.", "Netral"),
    ("Film ini berdurasi sekitar dua jam.", "Netral"),
    ("Toko ini buka mulai jam sembilan sampai jam lima sore.", "Netral"),
    ("Saya membeli laptop ini minggu lalu.", "Netral"),
    ("Aplikasi ini tersedia di Android dan iOS.", "Netral"),
    ("Besok ada rapat di kantor pukul dua siang.", "Netral"),
    ("Hotel ini terletak di dekat stasiun kereta.", "Netral"),
    ("Warna produknya hitam dan ukurannya sedang.", "Netral"),
    ("Kami menonton film itu di bioskop kemarin malam.", "Netral"),
    ("Kafe ini menyediakan kopi, teh, dan roti bakar.", "Netral"),
    ("Dosen memberikan tugas untuk dikumpulkan hari Jumat.", "Netral"),
    ("Pengiriman menggunakan jasa kurir reguler.", "Netral"),
    ("Hari ini hujan turun sejak pagi.", "Netral"),
    ("Saya baru saja pindah ke Bandung bulan ini.", "Netral"),
    ("Pertandingan dimulai pukul tujuh malam.", "Netral"),
    ("Buku ini terdiri dari dua belas bab.", "Netral"),
    ("Pesanan saya masih dalam proses pengemasan.", "Netral"),
    ("Aplikasi ini memiliki fitur pencarian dan notifikasi.", "Netral"),
    ("Nomor antrean saya adalah tiga puluh dua.", "Netral"),
    ("Kami tiba di lokasi wisata sekitar tengah hari.", "Netral"),
]
 
df = pd.DataFrame(DATASET, columns=["teks", "label"])
 
 
# ======================================================================
# 1b. DATASET BAHASA INGGRIS (untuk TextBlob)
# ======================================================================
DATASET_EN = [
    # ---------- POSITIVE: product ----------
    ("This product is amazing, works perfectly and arrived fast!", "Positive"),
    ("Excellent quality and the price is very reasonable.", "Positive"),
    ("The seller was friendly and responded quickly, will buy again.", "Positive"),
    ("Battery life is great and the screen looks beautiful.", "Positive"),
    ("Packaging was neat and the item arrived safely. Great job!", "Positive"),
    ("These shoes are comfortable to wear all day, perfect fit.", "Positive"),
    ("The service is very satisfying and fast!", "Positive"),
    # ---------- POSITIVE: movie ----------
    ("The movie was thrilling, the plot stayed interesting until the end.", "Positive"),
    ("The acting was outstanding, I was genuinely moved.", "Positive"),
    ("Beautiful cinematography and great music, highly recommended!", "Positive"),
    ("This film was very entertaining with an unexpected satisfying ending.", "Positive"),
    # ---------- POSITIVE: social media ----------
    ("So happy today, finally passed my thesis defense!", "Positive"),
    ("The food at this place is really delicious, huge portions too.", "Positive"),
    ("This new cafe is cozy and the service is great, must try.", "Positive"),
    ("The app is easy to use and looks really nice.", "Positive"),
    ("Last vacation was wonderful, the scenery was stunning.", "Positive"),
    ("The hotel was clean, staff friendly, breakfast was delicious. Very satisfied.", "Positive"),
 
    # ---------- NEGATIVE: product ----------
    ("This product is terrible, nothing like the photos. Disappointed!", "Negative"),
    ("Poor quality, it broke after just a week of use.", "Negative"),
    ("Shipping was super slow and the package arrived dented.", "Negative"),
    ("The seller was rude and took forever to reply.", "Negative"),
    ("Battery drains fast and the phone overheats, very disappointing.", "Negative"),
    ("These shoes are uncomfortable and run too small. Regret buying.", "Negative"),
    ("The service is very bad and slow!", "Negative"),
    # ---------- NEGATIVE: movie ----------
    ("The movie was boring, the story was slow and predictable.", "Negative"),
    ("The acting was stiff and the dialogue was bad, a waste of time.", "Negative"),
    ("The ending was disappointing, such a shame since it started well.", "Negative"),
    ("Worst movie I have ever watched, not recommended at all.", "Negative"),
    # ---------- NEGATIVE: social media ----------
    ("So annoyed, terrible traffic and I was late for work again.", "Negative"),
    ("The food was bland and overpriced, service was slow too.", "Negative"),
    ("The app keeps crashing and it's really frustrating.", "Negative"),
    ("The hotel was dirty, room smelled bad, staff didn't care. Very disappointed.", "Negative"),
    ("Customer service was completely unhelpful, truly terrible.", "Negative"),
 
    # ---------- NEUTRAL ----------
    ("The package was delivered on Monday at ten in the morning.", "Neutral"),
    ("This movie runs for about two hours.", "Neutral"),
    ("The store is open from nine to five.", "Neutral"),
    ("I bought this laptop last week.", "Neutral"),
    ("The app is available on Android and iOS.", "Neutral"),
    ("There is a meeting tomorrow at two in the afternoon.", "Neutral"),
    ("The hotel is located near the train station.", "Neutral"),
    ("The product comes in black and in a medium size.", "Neutral"),
    ("We watched that movie at the cinema last night.", "Neutral"),
    ("This cafe serves coffee, tea, and toast.", "Neutral"),
    ("The lecturer assigned homework due on Friday.", "Neutral"),
    ("Shipping is handled by a regular courier service.", "Neutral"),
    ("It has been raining since this morning.", "Neutral"),
    ("I just moved to a new city this month.", "Neutral"),
    ("The match starts at seven in the evening.", "Neutral"),
    ("This book has twelve chapters.", "Neutral"),
    ("My order is still being packed.", "Neutral"),
    ("The app has a search feature and notifications.", "Neutral"),
]
 
df_en = pd.DataFrame(DATASET_EN, columns=["text", "label"])
 
 
# ======================================================================
# 1c. GENERATOR KOMBINASI OTOMATIS (1.000 Positif / 1.000 Negatif / 1.000
#     Netral untuk Bahasa Indonesia, dan jumlah yang sama untuk English)
#
# Kalimat dibentuk dari kombinasi {subjek} x {frasa} x {alasan/detail}.
# Setiap kombinasi (subjek, frasa, alasan) bersifat unik, lalu diacak
# dengan seed tetap supaya hasilnya selalu sama tiap kali dijalankan,
# kemudian diambil 1.000 kombinasi pertama.
# ======================================================================
def buat_kombinasi(template: str, daftar_slot: list, label: str, n: int = 1000, seed: int = 42):
    """Menghasilkan n kalimat unik dari kombinasi beberapa daftar kata/frasa."""
    semua_kombinasi = list(itertools.product(*daftar_slot))
    rng = random.Random(seed)
    rng.shuffle(semua_kombinasi)
    if len(semua_kombinasi) < n:
        raise ValueError(f"Kombinasi tidak cukup: {len(semua_kombinasi)} < {n}. Tambah variasi slot.")
    hasil = []
    for kombinasi in semua_kombinasi[:n]:
        kalimat = template.format(*kombinasi)
        hasil.append((kalimat, label))
    return hasil
 
 
# ---------------------- Bahasa Indonesia: Positif ----------------------
ID_SUBJEK = [
    "Produk ini", "Film ini", "Pelayanan ini", "Aplikasi ini", "Restoran ini",
    "Hotel ini", "Driver ojek online ini", "Kualitas barang ini", "Tokonya",
    "Pengalaman belanja ini",
]
ID_POS_FRASA = [
    "sangat memuaskan", "benar-benar bagus", "luar biasa kerennya",
    "membuat saya senang", "melebihi ekspektasi saya", "sangat saya rekomendasikan",
    "sangat sepadan dengan harganya", "sangat berkualitas",
    "membuat hari saya menyenangkan", "sangat nyaman digunakan",
    "sangat cepat dan efisien",
]
ID_POS_ALASAN = [
    "pelayanannya ramah dan cepat", "kualitasnya sangat terjaga",
    "harganya juga terjangkau", "pengirimannya sangat cepat",
    "desainnya elegan dan modern", "fiturnya sangat lengkap",
    "rasanya enak sekali", "suasananya nyaman dan bersih",
    "stafnya sangat membantu", "semua sesuai dengan deskripsi",
]
DATASET_ID_POSITIF = buat_kombinasi(
    "{} {}, karena {}.", [ID_SUBJEK, ID_POS_FRASA, ID_POS_ALASAN], "Positif", n=1000, seed=1
)
 
# ---------------------- Bahasa Indonesia: Negatif ----------------------
ID_NEG_FRASA = [
    "sangat mengecewakan", "benar-benar buruk", "jauh dari ekspektasi",
    "membuat saya kesal", "tidak sesuai deskripsi", "sangat tidak memuaskan",
    "benar-benar buang-buang uang", "sangat buruk kualitasnya",
    "membuat hari saya buruk", "sangat merepotkan",
    "sangat lambat dan menyebalkan",
]
ID_NEG_ALASAN = [
    "pelayanannya lambat dan tidak ramah", "kualitasnya buruk dan cepat rusak",
    "harganya tidak sebanding dengan kualitasnya", "pengirimannya sangat lama",
    "banyak kesalahan yang terjadi", "fiturnya sering error",
    "rasanya tidak enak", "suasananya kotor dan berisik",
    "stafnya tidak sopan", "tidak sesuai dengan yang dijanjikan",
]
DATASET_ID_NEGATIF = buat_kombinasi(
    "{} {}, karena {}.", [ID_SUBJEK, ID_NEG_FRASA, ID_NEG_ALASAN], "Negatif", n=1000, seed=2
)
 
# ---------------------- Bahasa Indonesia: Netral ----------------------
ID_NEU_SUBJEK = [
    "Paket", "Film", "Toko", "Aplikasi", "Pertemuan",
    "Hotel", "Produk", "Acara", "Kereta", "Pesanan",
]
ID_NEU_AKSI = [
    "akan tiba", "berlangsung", "dibuka", "diperbarui", "dimulai",
    "tersedia", "dijadwalkan", "diproses", "berangkat", "selesai", "ditinjau",
]
ID_NEU_DETAIL = [
    "pada hari Senin", "pukul sepuluh pagi", "minggu depan",
    "di lokasi yang sama", "sesuai jadwal", "tanpa ada perubahan",
    "dengan rincian yang sudah ditentukan", "seperti biasanya",
    "pada bulan ini", "sesuai informasi terbaru",
]
DATASET_ID_NETRAL = buat_kombinasi(
    "{} {} {}.", [ID_NEU_SUBJEK, ID_NEU_AKSI, ID_NEU_DETAIL], "Netral", n=1000, seed=3
)
 
# ---------------------- English: Positive ----------------------
EN_SUBJECT = [
    "This product", "This movie", "This service", "This app", "This restaurant",
    "This hotel", "The driver", "The quality of this item", "This store",
    "This shopping experience",
]
EN_POS_PHRASE = [
    "is absolutely amazing", "is really great", "exceeded my expectations",
    "made me very happy", "is totally worth it", "is highly recommended",
    "is extremely satisfying", "works perfectly", "feels very premium",
    "is fast and efficient", "left me very impressed",
]
EN_POS_REASON = [
    "the service was friendly and quick", "the quality is excellent",
    "the price is also reasonable", "delivery was super fast",
    "the design is elegant and modern", "the features are complete",
    "it tastes really good", "the atmosphere is cozy and clean",
    "the staff were very helpful", "everything matched the description",
]
DATASET_EN_POSITIVE = buat_kombinasi(
    "{} {} because {}.", [EN_SUBJECT, EN_POS_PHRASE, EN_POS_REASON], "Positive", n=1000, seed=4
)
 
# ---------------------- English: Negative ----------------------
EN_NEG_PHRASE = [
    "is really disappointing", "is absolutely terrible", "fell far short of expectations",
    "made me upset", "doesn't match the description", "is extremely unsatisfying",
    "feels like a waste of money", "is very poor in quality",
    "ruined my day", "is very frustrating to deal with", "is slow and annoying",
]
EN_NEG_REASON = [
    "the service was slow and unfriendly", "the quality is poor and breaks easily",
    "the price doesn't match the quality", "delivery took way too long",
    "there were many mistakes", "the features keep crashing",
    "it doesn't taste good", "the atmosphere is dirty and noisy",
    "the staff were rude", "it didn't match what was promised",
]
DATASET_EN_NEGATIVE = buat_kombinasi(
    "{} {} because {}.", [EN_SUBJECT, EN_NEG_PHRASE, EN_NEG_REASON], "Negative", n=1000, seed=5
)
 
# ---------------------- English: Neutral ----------------------
EN_NEU_SUBJECT = [
    "The package", "The movie", "The store", "The app", "The meeting",
    "The hotel", "The product", "The event", "The train", "The order",
]
EN_NEU_ACTION = [
    "will arrive", "takes place", "opens", "gets updated", "starts",
    "is available", "is scheduled", "is being processed", "departs", "is completed", "is reviewed",
]
EN_NEU_DETAIL = [
    "on Monday", "at ten in the morning", "next week",
    "at the same location", "as planned", "with no changes",
    "according to the fixed details", "as usual",
    "this month", "based on the latest information",
]
DATASET_EN_NEUTRAL = buat_kombinasi(
    "{} {} {}.", [EN_NEU_SUBJECT, EN_NEU_ACTION, EN_NEU_DETAIL], "Neutral", n=1000, seed=6
)
 
# ---------------------- Gabungkan semua ke df dan df_en ----------------------
df = pd.concat(
    [
        df,
        pd.DataFrame(DATASET_ID_POSITIF, columns=["teks", "label"]),
        pd.DataFrame(DATASET_ID_NEGATIF, columns=["teks", "label"]),
        pd.DataFrame(DATASET_ID_NETRAL, columns=["teks", "label"]),
    ],
    ignore_index=True,
).sample(frac=1, random_state=0).reset_index(drop=True)  # diacak ulang
 
df_en = pd.concat(
    [
        df_en,
        pd.DataFrame(DATASET_EN_POSITIVE, columns=["text", "label"]),
        pd.DataFrame(DATASET_EN_NEGATIVE, columns=["text", "label"]),
        pd.DataFrame(DATASET_EN_NEUTRAL, columns=["text", "label"]),
    ],
    ignore_index=True,
).sample(frac=1, random_state=0).reset_index(drop=True)  # diacak ulang
 
 
# ======================================================================
# 2. ANALISIS TEXTBLOB (Bahasa Inggris)
# ======================================================================
TEXTBLOB_THRESHOLD = 0.1  # > 0.1 = Positive, < -0.1 = Negative, selain itu Neutral
 
 
def analisis_textblob(text: str, threshold: float = TEXTBLOB_THRESHOLD):
    """Mengembalikan (polaritas, subjektivitas, label) memakai TextBlob."""
    from textblob import TextBlob
 
    blob = TextBlob(text)
    polaritas = blob.sentiment.polarity
    subjektivitas = blob.sentiment.subjectivity
 
    if polaritas > threshold:
        label = "Positive"
    elif polaritas < -threshold:
        label = "Negative"
    else:
        label = "Neutral"
    return polaritas, subjektivitas, label
 
 
def demo_textblob():
    try:
        from textblob import TextBlob  # noqa: F401
    except ImportError:
        print("TextBlob belum terpasang. Jalankan: pip install textblob")
        return
 
    print("=" * 60)
    print("DEMO TEXTBLOB (dataset bahasa Inggris)")
    print("=" * 60)
    benar = 0
    for text, label_asli in DATASET_EN[:8]:
        pol, subj, label_prediksi = analisis_textblob(text)
        status = "OK" if label_prediksi == label_asli else "SALAH"
        benar += label_prediksi == label_asli
        print(f"[{status:5}] ({pol:+.2f}) {text[:55]:<55} -> {label_prediksi} (asli: {label_asli})")
    print()
 
 
# ======================================================================
# 3. ANALISIS BERBASIS KAMUS (Bahasa Indonesia) + THRESHOLD
# ======================================================================
KATA_POSITIF = {
    "bagus", "baik", "memuaskan", "puas", "suka", "senang", "seru", "menarik",
    "keren", "indah", "nyaman", "ramah", "cepat", "mantap", "murah", "terjangkau",
    "enak", "bersih", "rapi", "aman", "awet", "jernih", "luar", "biasa",
    "menghibur", "direkomendasikan", "recommended", "bangga", "semangat",
    "terharu", "sopan", "responsif", "menyenangkan", "wajib", "hebat",
    "sempurna", "lezat", "terima", "kasih", "mudah", "dipahami", "pas", "lulus",
}
KATA_NEGATIF = {
    "jelek", "buruk", "kecewa", "mengecewakan", "lambat", "lama", "rusak",
    "boros", "panas", "kesal", "sedih", "membosankan", "kaku", "hambar",
    "mahal", "kotor", "bau", "cuek", "galak", "error", "frustrasi", "macet",
    "parah", "menyesal", "penyok", "bosan", "susah", "semrawut", "emosi",
    "terburuk", "sayang", "telat", "kekecilan", "benci", "payah", "lemot",
    "buang", "tidak_membantu",
}
KATA_NEGASI = {"tidak", "tak", "bukan", "kurang", "belum", "nggak", "gak", "ga"}
KATA_PENGUAT = {"sangat", "banget", "sekali", "amat", "parah", "benar-benar", "terlalu"}
 
THRESHOLD = 0.1  # > 0.1 = Positif, < -0.1 = Negatif, selain itu Netral
 
 
def tokenisasi(teks: str):
    teks = teks.lower()
    return re.findall(r"[a-z]+(?:-[a-z]+)?", teks)
 
 
def hitung_polaritas(teks: str) -> float:
    """Mengembalikan skor polaritas antara -1.0 (negatif) dan 1.0 (positif)."""
    token = tokenisasi(teks)
    total, jumlah = 0.0, 0
 
    for i, kata in enumerate(token):
        if kata in KATA_POSITIF:
            nilai = 1.0
        elif kata in KATA_NEGATIF:
            nilai = -1.0
        else:
            continue
 
        # Cek penguat (kata sebelum atau sesudah)
        sekitar = token[max(0, i - 1): i] + token[i + 1: i + 2]
        if any(k in KATA_PENGUAT for k in sekitar):
            nilai *= 1.5
 
        # Cek negasi pada 2 kata sebelumnya ("tidak bagus" -> negatif)
        if any(k in KATA_NEGASI for k in token[max(0, i - 2): i]):
            nilai *= -1
 
        total += nilai
        jumlah += 1
 
    if jumlah == 0:
        return 0.0
    polaritas = total / (1.5 * jumlah)  # normalisasi ke rentang -1..1
    return max(-1.0, min(1.0, polaritas))
 
 
def tentukan_label(polaritas: float, threshold: float = THRESHOLD) -> str:
    if polaritas > threshold:
        return "Positif"
    if polaritas < -threshold:
        return "Negatif"
    return "Netral"
 
 
def evaluasi_textblob():
    print("=" * 60)
    print("EVALUASI TEXTBLOB PADA SELURUH DATASET BAHASA INGGRIS")
    print("=" * 60)
    hasil = df_en["text"].apply(lambda t: analisis_textblob(t))
    df_en["polaritas"] = hasil.apply(lambda h: h[0])
    df_en["prediksi"] = hasil.apply(lambda h: h[2])
 
    print(df_en[["text", "label", "polaritas", "prediksi"]]
          .head(12)
          .to_string(index=False, max_colwidth=45))
    print("...")
    print(f"\nAkurasi (TextBlob): {accuracy_score(df_en['label'], df_en['prediksi']):.2%}")
    print(classification_report(df_en["label"], df_en["prediksi"], zero_division=0))
 
 
def evaluasi_kamus():
    print("=" * 60)
    print("EVALUASI ANALISIS BERBASIS KAMUS + THRESHOLD")
    print("=" * 60)
    df["polaritas"] = df["teks"].apply(hitung_polaritas)
    df["prediksi"] = df["polaritas"].apply(tentukan_label)
 
    print(df[["teks", "label", "polaritas", "prediksi"]]
          .head(12)
          .to_string(index=False, max_colwidth=45))
    print("...")
    print(f"\nAkurasi (kamus): {accuracy_score(df['label'], df['prediksi']):.2%}")
    print(classification_report(df["label"], df["prediksi"], zero_division=0))
 
 
# ======================================================================
# 4. MODEL MACHINE LEARNING (Scikit-learn)
# ======================================================================
def latih_model_ml():
    print("=" * 60)
    print("MODEL SCIKIT-LEARN (TF-IDF + Logistic Regression)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        df["teks"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000),
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Akurasi pada data uji: {accuracy_score(y_test, pred):.2%}")
    print(classification_report(y_test, pred, zero_division=0))
    return model
 
 
# ======================================================================
# 5. MODE INTERAKTIF
# ======================================================================
def mode_interaktif(model):
    print("=" * 60)
    print("MODE INTERAKTIF (ketik 'keluar' untuk berhenti)")
    print("=" * 60)
    print("Pilih bahasa: [1] Indonesia  [2] English")
    pilihan = input("Pilihan (1/2): ").strip()
    bahasa_en = pilihan == "2"
 
    while True:
        teks = input("\nMasukkan kalimat: ").strip()
        if teks.lower() in {"keluar", "exit", "q"}:
            print("Sampai jumpa!")
            break
        if not teks:
            continue
 
        if bahasa_en:
            pol, subj, label = analisis_textblob(teks)
            print(f"[TextBlob] Polaritas: {pol:+.2f}  Subjektivitas: {subj:.2f} -> {label}")
        else:
            pol = hitung_polaritas(teks)
            print(f"[Kamus]        Polaritas: {pol:+.2f} -> {tentukan_label(pol)}")
            print(f"[Scikit-learn] Prediksi : {model.predict([teks])[0]}")
 
 
if __name__ == "__main__":
    print(f"Jumlah data Indonesia: {len(df)}")
    print(df["label"].value_counts().to_string(), "\n")
    print(f"Jumlah data English  : {len(df_en)}")
    print(df_en["label"].value_counts().to_string(), "\n")
 
    evaluasi_textblob()
    evaluasi_kamus()
    model_ml = latih_model_ml()
    mode_interaktif(model_ml)