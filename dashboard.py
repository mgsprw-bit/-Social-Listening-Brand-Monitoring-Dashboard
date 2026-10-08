"""
Dashboard Analisis Media Sosial — Social Listening & Brand Monitoring
=========================================================================
Jalankan dengan:  streamlit run dashboard.py
Satu folder dengan: analisis_sentimen.py, preprocessing.py, nlp_metrics.py,
                     ingestion.py, database.py, alerts.py, bias_check.py
"""

from datetime import datetime

import pandas as pd
import streamlit as st

import analisis_sentimen as inti
import database as db
import model_training as mt
from alerts import deteksi_krisis, kirim_webhook
from bias_check import (
    CATATAN_METODOLOGI,
    cek_keseimbangan_kelas,
    hitung_cakupan_deteksi,
    hitung_tingkat_ketidaksepakatan,
)
from ingestion import CATATAN_LEGAL, SKEMA_STANDAR, baca_file_upload, deteksi_pemetaan_otomatis
from nlp_metrics import deteksi_emosi, ekstrak_sentimen_aspek, hitung_nss, kata_terpopuler, pemodelan_topik
from preprocessing import preprocess_pipeline

LABEL_KE_STANDAR = {"Positive": "Positif", "Negative": "Negatif", "Neutral": "Netral"}
STANDAR_KE_LABEL_EN = {"Positif": "Positive", "Negatif": "Negative", "Netral": "Neutral"}

st.set_page_config(page_title="Social Listening Dashboard", page_icon="📡", layout="wide")

db.init_db()

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=SKEMA_STANDAR)
if "riwayat_nss" not in st.session_state:
    st.session_state.riwayat_nss = []  # list of (label_periode, nss)
if "pesan_hapus" not in st.session_state:
    st.session_state.pesan_hapus = ""
if "versi_editor_hapus" not in st.session_state:
    st.session_state.versi_editor_hapus = 0
if "versi_model" not in st.session_state:
    st.session_state.versi_model = {"id": 0, "en": 0}  # naik tiap kali model dilatih ulang


# ======================================================================
# FUNGSI ANALISIS (CACHED — pengganti ringan "Redis caching layer")
# ======================================================================
@st.cache_resource
def muat_model_ml(bahasa: str, _versi_cache: int = 0):
    """Memuat model dari disk (model_training.py). `_versi_cache` dipakai
    sebagai 'tombol pembersih cache' — naikkan nilainya (lewat
    st.cache_resource.clear() dipanggil dari luar) setiap selesai
    melatih ulang, supaya dashboard langsung memakai model terbaru."""
    return mt.muat_model(bahasa)


@st.cache_data(show_spinner=False)
def analisis_batch(daftar_teks: tuple, bahasa: str, _versi_model: int = 0):
    """Jalankan sentimen + emosi untuk seluruh dokumen. Di-cache supaya
    tidak dihitung ulang tiap kali widget lain di-interaksi (mensimulasikan
    manfaat caching layer seperti Redis, meski di sini lokal per-proses).
    `_versi_model` disertakan di key cache supaya cache otomatis basi
    (invalid) setelah model dilatih ulang."""
    baris = []
    model_ml = muat_model_ml(bahasa, _versi_model)
    for teks in daftar_teks:
        if bahasa == "id":
            pol = inti.hitung_polaritas(teks)
            label_kamus = inti.tentukan_label(pol)
        else:
            pol, _subj, label_kamus = inti.analisis_textblob(teks)
            label_kamus = LABEL_KE_STANDAR.get(label_kamus, label_kamus)  # standarkan ke Positif/Negatif/Netral
        label_ml = model_ml.predict([teks])[0]  # model sudah dilatih dengan label standar utk kedua bahasa
        emosi = deteksi_emosi(teks, bahasa)
        baris.append({
            "text": teks, "polaritas": pol, "label_kamus": label_kamus,
            "label_ml": label_ml, "emosi": emosi,
        })
    return pd.DataFrame(baris)


def simpan_batch_ke_db(df_baru: pd.DataFrame, bahasa: str):
    """Jalankan analisis sentimen untuk baris yang baru saja ditambahkan,
    lalu simpan ke SQLite — inilah yang membuat database bertambah
    otomatis setiap kali ada data baru masuk, bukan hanya tersimpan
    sementara di session_state."""
    if len(df_baru) == 0:
        return
    hasil = analisis_batch(tuple(df_baru["text"].astype(str)), bahasa, st.session_state.versi_model[bahasa])
    df_simpan = df_baru.copy().reset_index(drop=True)
    df_simpan["label_sentimen"] = hasil["label_kamus"].values
    df_simpan["polaritas"] = hasil["polaritas"].values
    df_simpan["emosi"] = hasil["emosi"].values
    db.simpan_posts(df_simpan)


def label_standar(label: str, bahasa: str) -> str:
    """Satukan label EN/ID ke satu set label standar untuk NSS & grafik."""
    peta = {"Positive": "Positif", "Negative": "Negatif", "Neutral": "Netral"}
    return peta.get(label, label) if bahasa == "en" else label


# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    bahasa_pilih = st.selectbox("Bahasa data", ["Bahasa Indonesia", "English"])
    bahasa_kode = "id" if bahasa_pilih == "Bahasa Indonesia" else "en"

    keyword_filter = st.text_input("Keyword / nama brand (label data ini)", value="brand_saya")

    st.divider()
    st.caption("Dataset saat ini di memori sesi:")
    st.metric("Jumlah dokumen", len(st.session_state.data))

    if st.button("🗑️ Kosongkan data sesi"):
        st.session_state.data = pd.DataFrame(columns=SKEMA_STANDAR)
        st.session_state.riwayat_nss = []
        st.rerun()


st.title("📡 Social Listening & Brand Monitoring Dashboard")
st.caption(
    "Mengubah komentar/ulasan/cuitan yang tidak terstruktur menjadi metrik terukur: "
    "volume, Net Sentiment Score, distribusi emosi, skor per aspek, dan topik populer."
)

(tab_input, tab_prep, tab_sentimen, tab_aspek, tab_topik, tab_tren,
 tab_krisis, tab_belajar, tab_metodologi) = st.tabs([
    "📥 Input Data", "🧹 Pra-pemrosesan", "📊 Sentimen & Ringkasan",
    "🎯 Aspek & Emosi", "☁️ Topik & Kata Kunci", "📈 Tren Waktu",
    "🚨 Peringatan Krisis", "🔁 Umpan Balik & Peningkatan Model", "ℹ️ Metodologi & Bias",
])


# ======================================================================
# TAB 1 — INPUT DATA (Batch Importer + template konektor API)
# ======================================================================
with tab_input:
    st.subheader("Unggah data (Batch Importer)")
    st.write("Format didukung: CSV, Excel (.xlsx), JSON. Minimal harus ada kolom berisi teks.")

    file_upload = st.file_uploader("Pilih file", type=["csv", "xlsx", "xls", "json"])

    if file_upload is not None:
        try:
            import io
            pratinjau_df = (
                pd.read_csv(file_upload) if file_upload.name.endswith(".csv")
                else pd.read_excel(file_upload) if file_upload.name.endswith((".xlsx", ".xls"))
                else pd.json_normalize(__import__("json").load(file_upload))
            )
            file_upload.seek(0)

            st.write("Pratinjau file (5 baris pertama):")
            st.dataframe(pratinjau_df.head(), use_container_width=True)

            pemetaan_otomatis = deteksi_pemetaan_otomatis(list(pratinjau_df.columns))
            st.write("Petakan kolom file Anda ke skema standar (sudah ditebak otomatis, silakan koreksi):")

            kolom_teks = st.selectbox(
                "Kolom mana yang berisi teks komentar/ulasan?",
                pratinjau_df.columns,
                index=list(pratinjau_df.columns).index(
                    [k for k, v in pemetaan_otomatis.items() if v == "text"][0]
                ) if any(v == "text" for v in pemetaan_otomatis.values()) else 0,
            )
            pemetaan_manual = {kolom_teks: "text"}
            for standar in ["timestamp", "likes", "shares", "comments", "author"]:
                kandidat_kolom = [k for k, v in pemetaan_otomatis.items() if v == standar]
                if kandidat_kolom:
                    pemetaan_manual[kandidat_kolom[0]] = standar

            if st.button("✅ Proses & Tambahkan ke Dataset", type="primary"):
                df_baru = baca_file_upload(file_upload, pemetaan_manual)
                df_baru["keyword"] = keyword_filter
                st.session_state.data = pd.concat([st.session_state.data, df_baru], ignore_index=True)
                with st.spinner("Menganalisis dan menyimpan ke database..."):
                    simpan_batch_ke_db(df_baru, bahasa_kode)
                st.success(f"{len(df_baru)} dokumen ditambahkan ke dataset sesi DAN disimpan permanen ke database.")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

    st.divider()
    st.subheader("Atau tempel teks langsung (uji cepat)")
    teks_manual = st.text_area("Satu kalimat per baris:", height=120,
                                placeholder="Pelayanan ini sangat memuaskan dan cepat!\nBarangnya jelek banget, kecewa.")
    if st.button("➕ Tambahkan teks ini ke dataset"):
        baris_manual = [b.strip() for b in teks_manual.split("\n") if b.strip()]
        if baris_manual:
            df_manual = pd.DataFrame({"text": baris_manual})
            for kolom in SKEMA_STANDAR:
                if kolom not in df_manual.columns:
                    df_manual[kolom] = 0 if kolom in ("likes", "shares", "comments") else ""
            df_manual["keyword"] = keyword_filter
            df_manual["timestamp"] = pd.Timestamp.now()
            st.session_state.data = pd.concat([st.session_state.data, df_manual], ignore_index=True)
            with st.spinner("Menganalisis dan menyimpan ke database..."):
                simpan_batch_ke_db(df_manual, bahasa_kode)
            st.success(f"{len(baris_manual)} baris ditambahkan ke sesi DAN disimpan permanen ke database.")

    st.divider()
    st.subheader("📤 Muat data yang sudah tersimpan di database")
    st.caption(f"Saat ini database berisi **{db.jumlah_posts_tersimpan():,} dokumen** secara total (lintas semua sesi).")
    if st.button("Muat dari database ke tampilan sesi ini"):
        df_dari_db = db.ambil_posts(keyword=keyword_filter)
        if len(df_dari_db):
            df_dari_db = df_dari_db.rename(columns={"label_sentimen": "label_sentimen_tersimpan"})
            kolom_dipakai = [k for k in SKEMA_STANDAR if k in df_dari_db.columns]
            st.session_state.data = pd.concat(
                [st.session_state.data, df_dari_db[kolom_dipakai]], ignore_index=True
            ).drop_duplicates(subset="text", keep="first")
            st.success(f"{len(df_dari_db)} dokumen dengan keyword '{keyword_filter}' dimuat dari database.")
        else:
            st.info(f"Belum ada data tersimpan untuk keyword '{keyword_filter}'.")

    # ------------------------------------------------------------------
    # KELOLA & HAPUS DATA DI DATABASE
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("🗑️ Kelola & hapus kalimat di database")
    st.caption("Centang kalimat yang ingin dihapus PERMANEN dari database. Tindakan ini tidak bisa dibatalkan.")

    if st.session_state.pesan_hapus:
        st.success(st.session_state.pesan_hapus)
        st.session_state.pesan_hapus = ""

    kolF1, kolF2 = st.columns([2, 1])
    with kolF1:
        kata_cari = st.text_input("Cari kata dalam kalimat (opsional)", key="cari_hapus")
    with kolF2:
        hanya_keyword_ini = st.checkbox(f"Hanya keyword '{keyword_filter}'", value=True, key="filter_kw_hapus")

    df_kelola = db.ambil_posts_dengan_id(
        keyword=keyword_filter if hanya_keyword_ini else None,
        cari_teks=kata_cari.strip() or None,
        batas=300,
    )

    if len(df_kelola) == 0:
        st.info("Tidak ada kalimat yang cocok di database.")
    else:
        df_kelola.insert(0, "hapus", False)
        versi_editor = st.session_state.versi_editor_hapus
        hasil_edit = st.data_editor(
            df_kelola,
            hide_index=True,
            use_container_width=True,
            disabled=[k for k in df_kelola.columns if k != "hapus"],
            column_config={"hapus": st.column_config.CheckboxColumn("Hapus?")},
            key=f"editor_hapus_{versi_editor}",  # kunci berganti setelah hapus agar centang lama tidak "bergeser"
        )
        baris_terpilih = hasil_edit[hasil_edit["hapus"]]
        id_terpilih = baris_terpilih["id"].astype(int).tolist()
        st.write(f"Kalimat terpilih: **{len(id_terpilih)}** (menampilkan maksimal 300 data terbaru)")

        yakin = st.checkbox("Saya yakin ingin menghapus permanen kalimat terpilih",
                            key=f"yakin_hapus_{versi_editor}")
        if st.button("🗑️ Hapus kalimat terpilih dari database", disabled=not (yakin and id_terpilih)):
            teks_dihapus = baris_terpilih["text"].tolist()
            jumlah_terhapus = db.hapus_posts(id_terpilih)
            # sinkronkan juga tampilan sesi supaya kalimat itu tidak ikut dianalisis lagi
            st.session_state.data = st.session_state.data[
                ~st.session_state.data["text"].isin(teks_dihapus)
            ].reset_index(drop=True)
            st.session_state.versi_editor_hapus += 1
            st.session_state.pesan_hapus = f"{jumlah_terhapus} kalimat dihapus permanen dari database."
            st.rerun()

    with st.expander("🔌 Konektor API Media Sosial (template, butuh API key Anda sendiri)"):
        st.code(
            'from ingestion import template_konektor_twitter_x\n'
            'df_baru = template_konektor_twitter_x(\n'
            '    bearer_token="ISI_TOKEN_ANDA",\n'
            '    kata_kunci="nama_brand_anda",\n'
            ')',
            language="python",
        )
        st.caption(CATATAN_LEGAL)


# ======================================================================
# TAB 2 — PRA-PEMROSESAN (transparan, bisa diaudit per tahap)
# ======================================================================
with tab_prep:
    st.subheader("Pratinjau Pipeline Pra-pemrosesan")
    if len(st.session_state.data) == 0:
        st.info("Belum ada data. Tambahkan data lewat tab 'Input Data' terlebih dahulu.")
    else:
        indeks_contoh = int(st.number_input(
            "Pilih nomor baris data untuk diperiksa (mulai dari 0)",
            min_value=0, max_value=max(len(st.session_state.data) - 1, 0), value=0, step=1,
        ))
        teks_contoh = str(st.session_state.data.iloc[indeks_contoh]["text"])
        hasil_pipeline = preprocess_pipeline(teks_contoh, bahasa_kode)

        for label_tahap, isi in hasil_pipeline.items():
            st.text(f"{label_tahap}:")
            st.code(str(isi))


# ======================================================================
# TAB 3 — SENTIMEN & RINGKASAN (kartu metrik + NSS)
# ======================================================================
with tab_sentimen:
    if len(st.session_state.data) == 0:
        st.info("Belum ada data. Tambahkan data lewat tab 'Input Data' terlebih dahulu.")
    else:
        with st.spinner("Menganalisis dokumen..."):
            hasil_analisis = analisis_batch(tuple(st.session_state.data["text"].astype(str)), bahasa_kode, st.session_state.versi_model[bahasa_kode])

        hasil_analisis["label_standar"] = hasil_analisis["label_kamus"].apply(lambda l: label_standar(l, bahasa_kode))

        volume = len(hasil_analisis)
        nss = hitung_nss(hasil_analisis["label_standar"])
        total_engagement = int(
            st.session_state.data[["likes", "shares", "comments"]].sum().sum()
        )

        kol1, kol2, kol3 = st.columns(3)
        kol1.metric("Total Volume Pembicaraan", f"{volume:,}")
        kol2.metric("Net Sentiment Score (NSS)", f"{nss:+.1f}", help="Rentang -100 (sangat negatif) s.d. +100 (sangat positif)")
        kol3.metric("Total Engagement", f"{total_engagement:,}", help="Jumlah likes + shares + comments")

        if st.button("📌 Catat NSS periode ini (untuk deteksi krisis)"):
            st.session_state.riwayat_nss.append((datetime.now().strftime("%Y-%m-%d %H:%M"), nss))
            st.success("NSS dicatat ke riwayat.")

        st.divider()
        kolA, kolB = st.columns(2)
        with kolA:
            st.write("**Distribusi Sentimen**")
            st.bar_chart(hasil_analisis["label_standar"].value_counts())
        with kolB:
            st.write("**Cek Keseimbangan Data (anti-bias)**")
            info_bias = cek_keseimbangan_kelas(hasil_analisis["label_standar"])
            (st.warning if not info_bias["seimbang"] else st.success)(info_bias["catatan"])

        st.divider()
        nama_metode_kamus = "Kamus" if bahasa_kode == "id" else "TextBlob"
        st.write(f"**Tingkat Ketidaksepakatan Metode {nama_metode_kamus} vs Scikit-learn**")
        info_disagree = hitung_tingkat_ketidaksepakatan(hasil_analisis["label_kamus"], hasil_analisis["label_ml"])
        kolX, kolY = st.columns([1, 2])
        kolX.metric("Sepakat", f"{info_disagree['persentase_sepakat']}%")
        kolY.caption(info_disagree["catatan"])

        with st.expander("Lihat data hasil analisis"):
            st.dataframe(hasil_analisis, use_container_width=True)


# ======================================================================
# TAB 4 — ASPEK & EMOSI
# ======================================================================
with tab_aspek:
    if len(st.session_state.data) == 0:
        st.info("Belum ada data.")
    else:
        hasil_analisis = analisis_batch(tuple(st.session_state.data["text"].astype(str)), bahasa_kode, st.session_state.versi_model[bahasa_kode])

        kolA, kolB = st.columns(2)
        with kolA:
            st.write("**Skor Sentimen per Aspek**")
            fungsi_pol = (lambda t: inti.hitung_polaritas(t)) if bahasa_kode == "id" else (lambda t: inti.analisis_textblob(t)[0])
            df_aspek = ekstrak_sentimen_aspek(list(hasil_analisis["text"]), fungsi_pol, bahasa_kode)
            df_aspek_valid = df_aspek.dropna(subset=["skor_rata_rata"])
            if len(df_aspek_valid):
                st.bar_chart(df_aspek_valid.set_index("aspek")["skor_rata_rata"])
                st.dataframe(df_aspek, use_container_width=True)
            else:
                st.caption("Tidak ada aspek yang terdeteksi secara eksplisit di data ini.")

        with kolB:
            st.write("**Distribusi Emosi**")
            distribusi_emosi = hasil_analisis["emosi"].value_counts()
            st.bar_chart(distribusi_emosi)

            info_cakupan = hitung_cakupan_deteksi(hasil_analisis["emosi"])
            (st.warning if info_cakupan["cakupan_persen"] < 60 else st.success)(
                f"Cakupan deteksi emosi: {info_cakupan['cakupan_persen']}%. {info_cakupan['catatan']}"
            )


# ======================================================================
# TAB 5 — TOPIK & WORD CLOUD
# ======================================================================
with tab_topik:
    if len(st.session_state.data) == 0:
        st.info("Belum ada data.")
    else:
        st.write("**Kata Paling Sering Muncul**")
        daftar_teks_bersih = [
            preprocess_pipeline(t, bahasa_kode)["teks_final"]
            for t in st.session_state.data["text"].astype(str)
        ]
        kata_populer = kata_terpopuler(st.session_state.data["text"].astype(str).tolist(), bahasa_kode, n=20)

        if kata_populer:
            df_kata = pd.DataFrame(kata_populer, columns=["kata", "frekuensi"])
            st.bar_chart(df_kata.set_index("kata"))

            try:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt

                wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(kata_populer))
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            except ImportError:
                st.caption("Install `wordcloud` (pip install wordcloud) untuk tampilan word cloud visual.")
        else:
            st.caption("Belum cukup data untuk menampilkan kata populer.")

        st.divider()
        st.write("**Pengelompokan Topik (LDA)**")
        n_topik = st.slider("Jumlah topik", 2, 8, 3)
        daftar_topik, pesan_error_topik = pemodelan_topik(
            st.session_state.data["text"].astype(str).tolist(), bahasa_kode, n_topik=n_topik
        )
        if pesan_error_topik:
            st.warning(pesan_error_topik)
        else:
            for topik in daftar_topik:
                st.write(f"**{topik['topik']}**: " + ", ".join(topik["kata_kunci"]))


# ======================================================================
# TAB 6 — TREN WAKTU
# ======================================================================
with tab_tren:
    if len(st.session_state.data) == 0:
        st.info("Belum ada data.")
    else:
        df_waktu = st.session_state.data.copy()
        df_waktu["timestamp"] = pd.to_datetime(df_waktu["timestamp"], errors="coerce")
        df_waktu = df_waktu.dropna(subset=["timestamp"])

        if len(df_waktu) == 0:
            st.info("Data tidak memiliki timestamp yang valid, sehingga tren waktu tidak dapat ditampilkan. "
                    "Pastikan kolom timestamp terisi saat upload data.")
        else:
            hasil_analisis = analisis_batch(tuple(st.session_state.data["text"].astype(str)), bahasa_kode, st.session_state.versi_model[bahasa_kode])
            df_waktu = df_waktu.reset_index(drop=True)
            df_waktu["label_standar"] = hasil_analisis["label_kamus"].apply(lambda l: label_standar(l, bahasa_kode))

            df_waktu["tanggal"] = df_waktu["timestamp"].dt.date
            volume_harian = df_waktu.groupby("tanggal").size()
            nss_harian = df_waktu.groupby("tanggal")["label_standar"].apply(hitung_nss)

            st.write("**Volume Pembicaraan per Hari**")
            st.line_chart(volume_harian)

            st.write("**NSS per Hari**")
            st.line_chart(nss_harian)


# ======================================================================
# TAB 7 — PERINGATAN KRISIS
# ======================================================================
with tab_krisis:
    st.subheader("Deteksi Lonjakan Sentimen Negatif")
    ambang = st.slider("Ambang batas penurunan NSS yang dianggap krisis (poin)", 5, 50, 20)

    if len(st.session_state.riwayat_nss) < 2:
        st.info("Catat NSS minimal 2 kali (dari tab 'Sentimen & Ringkasan') untuk mengaktifkan deteksi krisis.")
    else:
        (_, nss_sebelumnya) = st.session_state.riwayat_nss[-2]
        (waktu_sekarang, nss_sekarang) = st.session_state.riwayat_nss[-1]
        hasil_krisis = deteksi_krisis(nss_sekarang, nss_sebelumnya, ambang_selisih=ambang)

        if hasil_krisis.terjadi_krisis:
            st.error(hasil_krisis.pesan)
            db.catat_peringatan(keyword_filter, hasil_krisis.pesan, nss_sekarang)
        else:
            st.success(hasil_krisis.pesan)

    st.divider()
    st.write("**Riwayat NSS tercatat (sesi ini)**")
    if st.session_state.riwayat_nss:
        st.dataframe(pd.DataFrame(st.session_state.riwayat_nss, columns=["waktu", "nss"]), use_container_width=True)
        st.line_chart(pd.DataFrame(st.session_state.riwayat_nss, columns=["waktu", "nss"]).set_index("waktu"))

    st.divider()
    with st.expander("🔔 Konfigurasi Notifikasi Otomatis (opsional)"):
        url_webhook = st.text_input("Webhook URL (Slack/Discord/Teams)", type="password")
        if st.button("Kirim tes notifikasi") and url_webhook:
            try:
                kirim_webhook(url_webhook, {"text": "Tes notifikasi dari Social Listening Dashboard"})
                st.success("Notifikasi tes berhasil dikirim.")
            except Exception as e:
                st.error(f"Gagal mengirim: {e}")

    st.divider()
    st.write("**Log peringatan tersimpan di database**")
    st.dataframe(db.ambil_log_peringatan(), use_container_width=True)


# ======================================================================
# TAB 8 — UMPAN BALIK & PENINGKATAN MODEL (human-in-the-loop learning)
# ======================================================================
with tab_belajar:
    st.subheader("Bagaimana model menjadi lebih baik di sini")
    st.info(
        "Model **tidak** melatih ulang dirinya sendiri dari prediksinya sendiri — itu berisiko "
        "menciptakan *feedback loop* yang memperkuat kesalahan. Sebagai gantinya: Anda mengoreksi "
        "prediksi yang salah di bawah ini, koreksi itu tersimpan permanen, lalu model dilatih ulang "
        "dari data awal + seluruh koreksi manusia yang terkumpul. Model membaik, tapi arahnya "
        "dikendalikan manusia."
    )

    if len(st.session_state.data) == 0:
        st.caption("Tambahkan data lewat tab 'Input Data' untuk mulai memberi umpan balik.")
    else:
        hasil_analisis = analisis_batch(
            tuple(st.session_state.data["text"].astype(str)), bahasa_kode, st.session_state.versi_model[bahasa_kode]
        )

        st.write("**Koreksi prediksi yang salah**")
        indeks_umpan = int(st.number_input(
            "Pilih nomor dokumen (mulai dari 0)",
            min_value=0, max_value=max(len(hasil_analisis) - 1, 0), value=0, step=1,
            key="input_umpan_balik",
        ))
        baris_terpilih = hasil_analisis.iloc[indeks_umpan]

        st.text_area("Teks:", value=baris_terpilih["text"], height=80, disabled=True)
        kol1, kol2 = st.columns(2)
        kol1.metric("Prediksi Kamus/TextBlob", baris_terpilih["label_kamus"])
        kol2.metric("Prediksi Scikit-learn", baris_terpilih["label_ml"])

        label_benar = st.selectbox("Menurut Anda, label yang benar untuk teks ini:",
                                    ["Positif", "Netral", "Negatif"], key="pilih_label_benar")
        if st.button("💾 Simpan koreksi ini"):
            db.simpan_koreksi(baris_terpilih["text"], bahasa_kode, baris_terpilih["label_ml"], label_benar)
            st.success("Koreksi tersimpan. Akan dipakai saat model dilatih ulang.")

        st.divider()
        df_koreksi_pending = db.ambil_koreksi(bahasa_kode, hanya_belum_dipakai=True)
        st.write(f"**Koreksi manusia yang terkumpul (belum dipakai melatih): {len(df_koreksi_pending)}**")
        if len(df_koreksi_pending):
            st.dataframe(df_koreksi_pending[["text", "label_prediksi", "label_benar", "waktu"]],
                         use_container_width=True)

        st.divider()
        kolA, kolB = st.columns([1, 1])
        with kolA:
            if st.button("🔁 Latih Ulang Model Sekarang", type="primary"):
                with st.spinner("Melatih ulang model..."):
                    hasil_latih = mt.latih_ulang(bahasa_kode)
                if hasil_latih["berhasil"]:
                    st.session_state.versi_model[bahasa_kode] += 1  # otomatis membasikan cache
                    st.success(
                        f"Model baru: akurasi validasi {hasil_latih['akurasi_validasi']:.2%} "
                        f"dari {hasil_latih['jumlah_data_latih']:,} data "
                        f"({hasil_latih['jumlah_koreksi_manusia']} di antaranya koreksi manusia)."
                    )
                    st.rerun()
                else:
                    st.warning(hasil_latih["pesan"])

        with kolB:
            auto_latih = st.checkbox("Latih ulang otomatis saat koreksi pending ≥ N")
            ambang_koreksi = st.number_input("N", min_value=5, max_value=500, value=20, step=5, disabled=not auto_latih)
            if auto_latih and len(df_koreksi_pending) >= ambang_koreksi:
                with st.spinner("Ambang tercapai — melatih ulang otomatis..."):
                    hasil_latih = mt.latih_ulang(bahasa_kode)
                if hasil_latih["berhasil"]:
                    st.session_state.versi_model[bahasa_kode] += 1
                    st.success(f"Dilatih ulang otomatis (akurasi {hasil_latih['akurasi_validasi']:.2%}).")
                    st.rerun()
        st.caption(
            "Catatan soal kata 'otomatis' di atas: karena ini aplikasi Streamlit biasa (bukan "
            "proses server 24 jam), pengecekan ambang batas hanya berjalan saat dashboard dibuka/"
            "di-refresh — bukan di latar belakang sepanjang waktu. Untuk penjadwalan sungguhan "
            "24 jam (mis. latih ulang tiap malam jam 2), perlu proses terpisah seperti cron job "
            "atau GitHub Actions terjadwal yang memanggil `model_training.latih_ulang()`."
        )

        st.divider()
        st.write("**Riwayat Pelatihan Model (bukti model membaik seiring waktu)**")
        df_riwayat = db.ambil_riwayat_model(bahasa_kode)
        if len(df_riwayat):
            st.line_chart(df_riwayat.set_index("waktu")["akurasi_validasi"])
            st.caption("Akurasi validasi tiap kali model dilatih ulang.")
            st.dataframe(df_riwayat, use_container_width=True)
        else:
            st.caption("Belum ada riwayat pelatihan. Klik 'Latih Ulang Model Sekarang' untuk memulai.")


# ======================================================================
# TAB 9 — METODOLOGI & BIAS
# ======================================================================
with tab_metodologi:
    st.markdown(CATATAN_METODOLOGI)
