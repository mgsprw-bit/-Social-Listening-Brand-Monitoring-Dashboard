"""
Modul Sistem Peringatan Krisis (Tahap 6 Arsitektur)
=======================================================
Mendeteksi lonjakan sentimen negatif dibanding periode sebelumnya, lalu
(opsional) mengirim notifikasi lewat webhook atau email. Pengiriman
notifikasi TIDAK otomatis aktif — harus dikonfigurasi eksplisit oleh
pengguna (URL webhook / kredensial SMTP), supaya tidak ada pesan yang
terkirim tanpa sepengetahuan pengguna.
"""

from dataclasses import dataclass


@dataclass
class HasilDeteksiKrisis:
    terjadi_krisis: bool
    pesan: str
    nss_sekarang: float
    nss_sebelumnya: float | None
    selisih: float | None


def deteksi_krisis(nss_sekarang: float, nss_sebelumnya: float | None,
                    ambang_selisih: float = 20.0) -> HasilDeteksiKrisis:
    """Bandingkan NSS periode ini vs periode sebelumnya. Jika NSS turun
    lebih dari `ambang_selisih` poin, tandai sebagai potensi krisis.

    Catatan: threshold 20 poin adalah nilai awal yang wajar, TAPI harus
    disesuaikan dengan karakteristik brand/topik masing-masing — brand
    dengan volume kecil wajar punya fluktuasi NSS lebih liar, sehingga
    threshold tetap/statis bisa memicu alarm palsu (false alarm bias).
    """
    if nss_sebelumnya is None:
        return HasilDeteksiKrisis(False, "Belum ada data periode sebelumnya untuk dibandingkan.",
                                   nss_sekarang, None, None)

    selisih = nss_sekarang - nss_sebelumnya
    if selisih <= -ambang_selisih:
        pesan = (f"⚠️ NSS turun {abs(selisih):.1f} poin (dari {nss_sebelumnya:.1f} "
                 f"menjadi {nss_sekarang:.1f}). Indikasi lonjakan sentimen negatif.")
        return HasilDeteksiKrisis(True, pesan, nss_sekarang, nss_sebelumnya, selisih)

    return HasilDeteksiKrisis(False, "Tidak ada lonjakan signifikan.", nss_sekarang, nss_sebelumnya, selisih)


def kirim_webhook(url_webhook: str, payload: dict, timeout: int = 10):
    """Kirim notifikasi ke webhook (mis. Slack/Discord/Teams incoming
    webhook URL). Hanya dipanggil jika pengguna mengisi URL secara
    eksplisit di dashboard — tidak pernah otomatis mengirim data keluar
    tanpa konfigurasi pengguna."""
    import requests
    respons = requests.post(url_webhook, json=payload, timeout=timeout)
    respons.raise_for_status()
    return respons.status_code


def kirim_email(smtp_host: str, smtp_port: int, pengirim: str, sandi_aplikasi: str,
                 penerima: str, subjek: str, isi: str):
    """Kirim email peringatan lewat SMTP. Gunakan App Password, BUKAN
    password akun utama, dan simpan kredensial di environment variable/
    secrets manager, jangan hardcode di kode yang di-commit ke GitHub."""
    import smtplib
    from email.mime.text import MIMEText

    pesan = MIMEText(isi)
    pesan["Subject"] = subjek
    pesan["From"] = pengirim
    pesan["To"] = penerima

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(pengirim, sandi_aplikasi)
        server.sendmail(pengirim, [penerima], pesan.as_string())
