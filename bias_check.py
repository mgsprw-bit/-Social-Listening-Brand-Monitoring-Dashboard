"""
Modul Pengecekan Bias & Transparansi Metodologi
===================================================
Kumpulan fungsi untuk MEMPERLIHATKAN keterbatasan hasil analisis kepada
pengguna, alih-alih menyembunyikannya. Prinsipnya: lebih baik dashboard
bilang "saya kurang yakin di sini" daripada memberi angka yang terlihat
presisi tapi menyesatkan.
"""

import pandas as pd


def cek_keseimbangan_kelas(label_series: pd.Series) -> dict:
    """Jika data yang diupload sangat condong ke satu sentimen (misalnya
    95% positif), model/metrik yang dihitung darinya bisa menyesatkan
    (mis. karena hanya mewakili pelanggan yang puas yang mau menulis
    ulasan). Fungsi ini memberi peringatan eksplisit soal itu."""
    if len(label_series) == 0:
        return {"seimbang": True, "catatan": "Tidak ada data."}

    proporsi = label_series.value_counts(normalize=True)
    label_dominan = proporsi.idxmax()
    persentase_dominan = proporsi.max() * 100

    if persentase_dominan > 80:
        catatan = (f"⚠️ {persentase_dominan:.1f}% data didominasi label '{label_dominan}'. "
                   "Hasil agregat (NSS, dsb.) kemungkinan besar bias ke arah itu — "
                   "pertimbangkan apakah cara pengumpulan data sudah representatif "
                   "(contoh: survei kepuasan biasanya under-sample pelanggan yang diam saja).")
        return {"seimbang": False, "catatan": catatan, "proporsi": proporsi.to_dict()}

    return {"seimbang": True, "catatan": "Distribusi label cukup seimbang.", "proporsi": proporsi.to_dict()}


def hitung_tingkat_ketidaksepakatan(label_kamus: pd.Series, label_ml: pd.Series) -> dict:
    """Bandingkan prediksi dua metode berbeda (kamus vs machine learning).
    Tingkat ketidaksepakatan yang tinggi adalah sinyal bahwa teks-teks
    tersebut ambigu/sulit, dan hasil otomatis sebaiknya ditinjau manual,
    bukan diambil sebagai kebenaran mutlak dari salah satu metode saja."""
    cocok = (label_kamus.values == label_ml.values)
    tingkat_sepakat = cocok.mean() * 100 if len(cocok) else 100.0
    return {
        "persentase_sepakat": round(tingkat_sepakat, 2),
        "persentase_tidak_sepakat": round(100 - tingkat_sepakat, 2),
        "catatan": (
            "Semakin rendah persentase sepakat, semakin besar porsi teks yang "
            "ambigu bagi sistem otomatis — pertimbangkan tinjauan manual pada "
            "sampel yang tidak sepakat ini, terutama sebelum mengambil keputusan "
            "bisnis penting (mis. respons krisis)."
        ),
    }


def hitung_cakupan_deteksi(label_series: pd.Series, label_tidak_terdeteksi: str = "Tidak terdeteksi") -> dict:
    """Berapa persen dokumen yang BERHASIL diberi label (emosi/aspek),
    bukan 'Tidak terdeteksi'. Cakupan rendah berarti kamus kata kunci
    belum cukup kaya untuk domain data ini — angka ringkasan (mis. %
    masing-masing emosi) sebaiknya dibaca dengan hati-hati kalau cakupan
    rendah, karena hanya mewakili sebagian kecil data."""
    total = len(label_series)
    if total == 0:
        return {"cakupan_persen": 0.0, "catatan": "Tidak ada data."}
    terdeteksi = (label_series != label_tidak_terdeteksi).sum()
    cakupan = terdeteksi / total * 100
    catatan = "Cakupan memadai." if cakupan >= 60 else (
        "⚠️ Cakupan deteksi rendah — banyak dokumen tidak mengandung kata kunci "
        "yang dikenali kamus. Pertimbangkan memperkaya kamus kata kunci sesuai "
        "istilah khas domain/industri Anda."
    )
    return {"cakupan_persen": round(cakupan, 2), "catatan": catatan}


CATATAN_METODOLOGI = """
### Metodologi & Keterbatasan

**Analisis sentimen (metode kamus & Scikit-learn)**
Dilatih/dibangun dari kombinasi contoh kalimat yang disusun untuk demonstrasi.
Untuk pemakaian produksi, latih ulang model dengan data berlabel ASLI dari
domain Anda (ulasan asli, bukan kalimat buatan), idealnya berimbang jumlah
tiap kelasnya dan divalidasi oleh lebih dari satu penilai manusia (inter-
annotator agreement) untuk mengurangi bias pelabelan.

**Deteksi emosi & Aspect-Based Sentiment**
Berbasis pencocokan kata kunci (lexicon-based), bukan model yang dilatih
dari data beremosi berlabel. Kelebihannya: transparan, bisa diaudit kata
per kata. Kekurangannya: tidak menangkap sarkasme, konteks tersirat, atau
istilah baru/gaul yang belum ada di kamus.

**Topic Modeling (LDA)**
Berbasis kemunculan kata bersama (bag-of-words), bukan makna semantik.
Topik yang dihasilkan bisa jadi sulit diinterpretasi kalau data sedikit
atau sangat beragam temanya.

**Rekomendasi umum untuk menghindari bias**
1. Jangan jadikan skor otomatis sebagai satu-satunya dasar keputusan besar
   (krisis komunikasi, kebijakan) — silangkan dengan tinjauan manusia atas
   sampel acak.
2. Periksa keseimbangan kelas data Anda (lihat kartu "Cek Bias Data" di
   dashboard) — data yang sangat timpang bisa membuat metrik menyesatkan.
3. Perhatikan tingkat ketidaksepakatan antar-metode sebagai sinyal
   keraguan sistem, bukan sesuatu yang perlu "dipilih salah satu saja".
4. Perkaya terus kamus kata kunci (slang, aspek, emosi) sesuai bahasa dan
   istilah khas audiens/industri Anda — kamus generik tidak akan pernah
   cocok sempurna untuk semua konteks.
"""
