"""
Modul Basis Data (Tahap 4 Arsitektur)
========================================
Memakai SQLite (bawaan Python, tanpa instalasi server terpisah) sebagai
pengganti ringan PostgreSQL/MongoDB — cocok untuk skala kecil-menengah
(ribuan-jutaan baris). Struktur tabel & index dibuat semirip mungkin
dengan desain yang diminta, sehingga mudah dipindah ke Postgres/Mongo
nanti kalau skala datanya sudah sangat besar (tinggal ganti fungsi
`get_koneksi()` dengan driver Postgres/Mongo, query SQL-nya kompatibel).
"""

import json
import sqlite3
from datetime import datetime

import pandas as pd

PATH_DB_DEFAULT = "medsos_monitoring.db"


def get_koneksi(path_db: str = PATH_DB_DEFAULT):
    return sqlite3.connect(path_db, check_same_thread=False)


def init_db(path_db: str = PATH_DB_DEFAULT):
    """Buat tabel transaksional (raw post) dan tabel agregasi time-series,
    lengkap dengan index pada keyword, timestamp, dan aspect."""
    konn = get_koneksi(path_db)
    kursor = konn.cursor()

    kursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            author TEXT,
            timestamp TEXT,
            likes INTEGER DEFAULT 0,
            shares INTEGER DEFAULT 0,
            comments INTEGER DEFAULT 0,
            keyword TEXT,
            source TEXT,
            label_sentimen TEXT,
            polaritas REAL,
            emosi TEXT,
            inserted_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    kursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_keyword ON posts(keyword)")
    kursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_timestamp ON posts(timestamp)")

    kursor.execute("""
        CREATE TABLE IF NOT EXISTS agregasi_harian (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tanggal TEXT,
            keyword TEXT,
            volume INTEGER,
            nss REAL,
            total_engagement INTEGER,
            distribusi_emosi TEXT,
            dihitung_pada TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    kursor.execute("CREATE INDEX IF NOT EXISTS idx_agregasi_tanggal ON agregasi_harian(tanggal)")
    kursor.execute("CREATE INDEX IF NOT EXISTS idx_agregasi_keyword ON agregasi_harian(keyword)")

    kursor.execute("""
        CREATE TABLE IF NOT EXISTS skor_aspek (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tanggal TEXT,
            keyword TEXT,
            aspect TEXT,
            skor_rata_rata REAL,
            jumlah_disebut INTEGER
        )
    """)
    kursor.execute("CREATE INDEX IF NOT EXISTS idx_aspek_aspect ON skor_aspek(aspect)")
    kursor.execute("CREATE INDEX IF NOT EXISTS idx_aspek_tanggal ON skor_aspek(tanggal)")

    kursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback_koreksi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            bahasa TEXT,
            label_prediksi TEXT,
            label_benar TEXT,
            dipakai_untuk_latih INTEGER DEFAULT 0,
            waktu TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    kursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_bahasa ON feedback_koreksi(bahasa)")

    kursor.execute("""
        CREATE TABLE IF NOT EXISTS riwayat_model (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bahasa TEXT,
            jumlah_data_latih INTEGER,
            jumlah_koreksi_manusia INTEGER,
            akurasi_validasi REAL,
            waktu TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    kursor.execute("""
        CREATE TABLE IF NOT EXISTS log_peringatan (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            waktu TEXT DEFAULT CURRENT_TIMESTAMP,
            keyword TEXT,
            pesan TEXT,
            nss_saat_itu REAL
        )
    """)

    konn.commit()
    konn.close()


def simpan_posts(df: pd.DataFrame, path_db: str = PATH_DB_DEFAULT):
    """Simpan dataframe post (hasil ingestion + analisis sentimen) ke
    tabel transaksional `posts`."""
    konn = get_koneksi(path_db)
    kolom_tabel = ["text", "author", "timestamp", "likes", "shares", "comments",
                   "keyword", "source", "label_sentimen", "polaritas", "emosi"]
    df_simpan = df.copy()
    for k in kolom_tabel:
        if k not in df_simpan.columns:
            df_simpan[k] = None
    df_simpan["timestamp"] = df_simpan["timestamp"].astype(str)
    df_simpan[kolom_tabel].to_sql("posts", konn, if_exists="append", index=False)
    konn.commit()
    konn.close()


def simpan_agregasi_harian(tanggal: str, keyword: str, volume: int, nss: float,
                            total_engagement: int, distribusi_emosi: dict,
                            path_db: str = PATH_DB_DEFAULT):
    konn = get_koneksi(path_db)
    konn.execute(
        "INSERT INTO agregasi_harian (tanggal, keyword, volume, nss, total_engagement, distribusi_emosi) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (tanggal, keyword, volume, nss, total_engagement, json.dumps(distribusi_emosi)),
    )
    konn.commit()
    konn.close()


def simpan_skor_aspek(tanggal: str, keyword: str, df_aspek: pd.DataFrame, path_db: str = PATH_DB_DEFAULT):
    konn = get_koneksi(path_db)
    for _, baris in df_aspek.iterrows():
        konn.execute(
            "INSERT INTO skor_aspek (tanggal, keyword, aspect, skor_rata_rata, jumlah_disebut) "
            "VALUES (?, ?, ?, ?, ?)",
            (tanggal, keyword, baris["aspek"], baris["skor_rata_rata"], baris["jumlah_disebut"]),
        )
    konn.commit()
    konn.close()


def catat_peringatan(keyword: str, pesan: str, nss_saat_itu: float, path_db: str = PATH_DB_DEFAULT):
    konn = get_koneksi(path_db)
    konn.execute(
        "INSERT INTO log_peringatan (keyword, pesan, nss_saat_itu) VALUES (?, ?, ?)",
        (keyword, pesan, nss_saat_itu),
    )
    konn.commit()
    konn.close()


def ambil_posts(keyword: str | None = None, path_db: str = PATH_DB_DEFAULT) -> pd.DataFrame:
    konn = get_koneksi(path_db)
    if keyword:
        df = pd.read_sql("SELECT * FROM posts WHERE keyword = ?", konn, params=(keyword,))
    else:
        df = pd.read_sql("SELECT * FROM posts", konn)
    konn.close()
    return df


def ambil_agregasi_harian(keyword: str | None = None, path_db: str = PATH_DB_DEFAULT) -> pd.DataFrame:
    konn = get_koneksi(path_db)
    if keyword:
        df = pd.read_sql(
            "SELECT * FROM agregasi_harian WHERE keyword = ? ORDER BY tanggal", konn, params=(keyword,)
        )
    else:
        df = pd.read_sql("SELECT * FROM agregasi_harian ORDER BY tanggal", konn)
    konn.close()
    return df


def ambil_log_peringatan(path_db: str = PATH_DB_DEFAULT) -> pd.DataFrame:
    konn = get_koneksi(path_db)
    df = pd.read_sql("SELECT * FROM log_peringatan ORDER BY waktu DESC", konn)
    konn.close()
    return df


def ambil_posts_dengan_id(keyword: str | None = None, cari_teks: str | None = None,
                           batas: int = 300, path_db: str = PATH_DB_DEFAULT) -> pd.DataFrame:
    """Ambil daftar kalimat beserta id-nya (untuk fitur kelola/hapus data).
    Dibatasi `batas` baris terbaru supaya tabel tetap ringan."""
    kondisi, parameter = [], []
    if keyword:
        kondisi.append("keyword = ?")
        parameter.append(keyword)
    if cari_teks:
        kondisi.append("text LIKE ?")
        parameter.append(f"%{cari_teks}%")
    where = ("WHERE " + " AND ".join(kondisi)) if kondisi else ""

    konn = get_koneksi(path_db)
    df = pd.read_sql(
        f"SELECT id, text, keyword, timestamp, label_sentimen FROM posts {where} ORDER BY id DESC LIMIT ?",
        konn, params=parameter + [int(batas)],
    )
    konn.close()
    return df


def hapus_posts(id_list: list, path_db: str = PATH_DB_DEFAULT) -> int:
    """Hapus PERMANEN baris posts berdasarkan daftar id. Mengembalikan
    jumlah baris yang benar-benar terhapus. Tidak bisa dibatalkan."""
    if not id_list:
        return 0
    konn = get_koneksi(path_db)
    placeholder = ",".join("?" * len(id_list))
    kursor = konn.execute(f"DELETE FROM posts WHERE id IN ({placeholder})", [int(i) for i in id_list])
    konn.commit()
    jumlah = kursor.rowcount
    konn.close()
    return jumlah


def jumlah_posts_tersimpan(path_db: str = PATH_DB_DEFAULT) -> int:
    konn = get_koneksi(path_db)
    n = konn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    konn.close()
    return n


# ======================================================================
# FEEDBACK MANUSIA (untuk peningkatan model yang aman, anti feedback-loop)
# ======================================================================
def simpan_koreksi(text: str, bahasa: str, label_prediksi: str, label_benar: str,
                    path_db: str = PATH_DB_DEFAULT):
    konn = get_koneksi(path_db)
    konn.execute(
        "INSERT INTO feedback_koreksi (text, bahasa, label_prediksi, label_benar) VALUES (?, ?, ?, ?)",
        (text, bahasa, label_prediksi, label_benar),
    )
    konn.commit()
    konn.close()


def ambil_koreksi(bahasa: str, hanya_belum_dipakai: bool = False, path_db: str = PATH_DB_DEFAULT) -> pd.DataFrame:
    konn = get_koneksi(path_db)
    query = "SELECT * FROM feedback_koreksi WHERE bahasa = ?"
    if hanya_belum_dipakai:
        query += " AND dipakai_untuk_latih = 0"
    df = pd.read_sql(query, konn, params=(bahasa,))
    konn.close()
    return df


def tandai_koreksi_terpakai(id_list: list, path_db: str = PATH_DB_DEFAULT):
    if not id_list:
        return
    konn = get_koneksi(path_db)
    placeholder = ",".join("?" * len(id_list))
    konn.execute(f"UPDATE feedback_koreksi SET dipakai_untuk_latih = 1 WHERE id IN ({placeholder})", id_list)
    konn.commit()
    konn.close()


def simpan_riwayat_model(bahasa: str, jumlah_data_latih: int, jumlah_koreksi_manusia: int,
                          akurasi_validasi: float, path_db: str = PATH_DB_DEFAULT):
    konn = get_koneksi(path_db)
    konn.execute(
        "INSERT INTO riwayat_model (bahasa, jumlah_data_latih, jumlah_koreksi_manusia, akurasi_validasi) "
        "VALUES (?, ?, ?, ?)",
        (bahasa, jumlah_data_latih, jumlah_koreksi_manusia, akurasi_validasi),
    )
    konn.commit()
    konn.close()


def ambil_riwayat_model(bahasa: str, path_db: str = PATH_DB_DEFAULT) -> pd.DataFrame:
    konn = get_koneksi(path_db)
    df = pd.read_sql("SELECT * FROM riwayat_model WHERE bahasa = ? ORDER BY waktu", konn, params=(bahasa,))
    konn.close()
    return df
