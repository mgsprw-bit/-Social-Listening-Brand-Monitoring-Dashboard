"""
Modul Data Ingestion (Tahap 1 Arsitektur)
===========================================
- Batch Importer: baca file CSV/Excel/JSON yang diunggah pengguna.
- Template Konektor API: kerangka kode untuk menarik data dari API resmi
  platform media sosial. TIDAK dijalankan otomatis — butuh API key/token
  milik pengguna sendiri, dan harus mematuhi Syarat Layanan (ToS) tiap
  platform. Kode scraping tanpa izin platform TIDAK disediakan di sini,
  karena umumnya melanggar ToS dan berisiko hukum.
"""

import json
import io
import pandas as pd

# Skema kolom standar yang dipakai di seluruh pipeline
SKEMA_STANDAR = ["text", "author", "timestamp", "likes", "shares", "comments", "keyword", "source"]


def baca_file_upload(file_upload, pemetaan_kolom: dict | None = None) -> pd.DataFrame:
    """Baca file CSV / Excel / JSON yang diunggah lewat Streamlit `file_uploader`,
    lalu normalisasi ke skema standar. `pemetaan_kolom` opsional untuk
    memetakan nama kolom asli -> nama kolom standar, misalnya:
        {"isi_komentar": "text", "tanggal": "timestamp"}
    """
    nama = file_upload.name.lower()

    if nama.endswith(".csv"):
        df = pd.read_csv(file_upload)
    elif nama.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_upload)
    elif nama.endswith(".json"):
        konten = json.load(file_upload)
        df = pd.json_normalize(konten if isinstance(konten, list) else [konten])
    else:
        raise ValueError("Format file tidak didukung. Gunakan CSV, Excel (.xlsx), atau JSON.")

    if pemetaan_kolom:
        df = df.rename(columns=pemetaan_kolom)

    # Pastikan kolom wajib ada; isi dengan nilai kosong/relevan jika tidak ada
    for kolom in SKEMA_STANDAR:
        if kolom not in df.columns:
            if kolom in ("likes", "shares", "comments"):
                df[kolom] = 0
            elif kolom == "timestamp":
                df[kolom] = pd.NaT
            else:
                df[kolom] = ""

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for kolom in ("likes", "shares", "comments"):
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce").fillna(0).astype(int)

    return df[SKEMA_STANDAR].copy()


def deteksi_pemetaan_otomatis(kolom_asli: list) -> dict:
    """Coba tebak pemetaan kolom berdasarkan nama yang umum dipakai,
    supaya pengguna tidak harus memetakan manual kalau nama kolomnya
    sudah mirip standar. Pengguna tetap bisa mengoreksi di UI."""
    kandidat = {
        "text": ["text", "teks", "komentar", "isi", "content", "caption", "tweet", "review", "ulasan"],
        "author": ["author", "user", "username", "penulis", "nama"],
        "timestamp": ["timestamp", "tanggal", "date", "created_at", "waktu"],
        "likes": ["likes", "like", "suka"],
        "shares": ["shares", "share", "retweet", "repost"],
        "comments": ["comments", "comment", "replies", "balasan"],
        "keyword": ["keyword", "kata_kunci", "tag", "hashtag"],
        "source": ["source", "platform", "sumber"],
    }
    pemetaan = {}
    kolom_lower = {k.lower(): k for k in kolom_asli}
    for standar, alias_list in kandidat.items():
        for alias in alias_list:
            if alias in kolom_lower:
                pemetaan[kolom_lower[alias]] = standar
                break
    return pemetaan


# ======================================================================
# TEMPLATE KONEKTOR API (contoh kerangka — perlu API key milik sendiri)
# ======================================================================
def template_konektor_twitter_x(bearer_token: str, kata_kunci: str, max_hasil: int = 100):
    """
    Contoh kerangka integrasi dengan X (Twitter) API v2.
    INI TIDAK DIJALANKAN OTOMATIS. Isi `bearer_token` dengan token dari
    developer.x.com milik akun Anda sendiri, lalu uncomment kode di bawah.

    Dokumentasi resmi: https://developer.x.com/en/docs/x-api
    """
    raise NotImplementedError(
        "Isi bearer_token dari akun developer X Anda sendiri, lalu "
        "uncomment kode contoh di dalam fungsi ini sebelum memakainya."
    )

    # import requests
    # url = "https://api.x.com/2/tweets/search/recent"
    # headers = {"Authorization": f"Bearer {bearer_token}"}
    # params = {
    #     "query": kata_kunci,
    #     "max_results": max_hasil,
    #     "tweet.fields": "created_at,public_metrics,author_id",
    # }
    # respons = requests.get(url, headers=headers, params=params, timeout=15)
    # respons.raise_for_status()
    # data = respons.json().get("data", [])
    # baris = []
    # for item in data:
    #     metrik = item.get("public_metrics", {})
    #     baris.append({
    #         "text": item.get("text", ""),
    #         "author": item.get("author_id", ""),
    #         "timestamp": item.get("created_at", ""),
    #         "likes": metrik.get("like_count", 0),
    #         "shares": metrik.get("retweet_count", 0),
    #         "comments": metrik.get("reply_count", 0),
    #         "keyword": kata_kunci,
    #         "source": "X (Twitter)",
    #     })
    # return pd.DataFrame(baris)


def template_konektor_instagram_graph(access_token: str, hashtag_id: str):
    """
    Contoh kerangka integrasi dengan Instagram Graph API (butuh akun
    Instagram Business/Creator yang terhubung ke Meta Developer App).
    Dokumentasi: https://developers.facebook.com/docs/instagram-api
    """
    raise NotImplementedError(
        "Isi access_token dan hashtag_id dari akun Meta Developer Anda "
        "sendiri, lalu uncomment kode contoh di dalam fungsi ini."
    )


CATATAN_LEGAL = """
PENTING soal pengumpulan data media sosial:
- Gunakan API RESMI tiap platform (X/Twitter API, Meta Graph API, dsb),
  bukan scraping tanpa izin — mayoritas platform melarang scraping di
  Syarat Layanan mereka dan dapat memblokir akun/IP Anda.
- Simpan API key/token di environment variable atau secrets manager,
  JANGAN ditulis langsung di kode yang di-commit ke GitHub publik.
- Patuhi batas rate limit resmi tiap API untuk menghindari pemblokiran.
"""
