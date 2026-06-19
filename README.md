# Dashboard Prediksi TPT Provinsi Jawa Barat

Dashboard interaktif Streamlit untuk analisis & prediksi Tingkat Pengangguran Terbuka (TPT)
27 kabupaten/kota di Jawa Barat (2018–2025), memakai **Regresi Linear Berganda** dan
**Random Forest Regressor**.

## 📁 Struktur Folder

```
tpt_dashboard/
├── app.py                     # Aplikasi utama Streamlit
├── requirements.txt           # Daftar dependency
├── .streamlit/
│   └── config.toml            # Tema warna (pink pastel, ungu pastel, biru)
├── data/
│   ├── data_integrated.csv    # Data gabungan 2018-2025 (27 wilayah x 8 tahun)
│   ├── data_train_80_20.csv / data_test_80_20.csv
│   ├── data_train_70_30.csv / data_test_70_30.csv
│   └── data_train_60_40.csv / data_test_60_40.csv
└── geo/
    └── jabar.geojson          # Batas wilayah 27 kabupaten/kota Jawa Barat
```

⚠️ **Penting:** nama kolom `Kabupaten_Kota` di file CSV harus persis sama formatnya
dengan properti `Kabupaten_Kota` di `geo/jabar.geojson` (contoh: `"Kabupaten Bandung"`,
`"Kota Bandung"`) agar peta choropleth tampil dengan benar.

## 🚀 Cara Menjalankan di Lokal

1. Pastikan Python 3.9–3.12 terpasang.
2. Buat virtual environment (opsional tapi disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```
5. Browser otomatis terbuka di `http://localhost:8501`.

## ☁️ Deploy ke Streamlit Community Cloud (Gratis)

1. **Buat akun GitHub** (jika belum punya) di https://github.com.
2. **Buat repository baru** (boleh public/private), misalnya `tpt-jabar-dashboard`.
3. **Upload semua isi folder `tpt_dashboard/`** ke repo tersebut — pastikan struktur
   folder (termasuk `data/`, `geo/`, `.streamlit/`) ikut ter-upload, bukan cuma `app.py`.
   - Cara termudah lewat web: GitHub → repo baru → "uploading an existing file" →
     drag & drop semua file/folder.
   - Atau lewat terminal:
     ```bash
     cd tpt_dashboard
     git init
     git add .
     git commit -m "Initial commit - dashboard TPT Jawa Barat"
     git branch -M main
     git remote add origin https://github.com/USERNAME/tpt-jabar-dashboard.git
     git push -u origin main
     ```
4. **Buka https://share.streamlit.io** (Streamlit Community Cloud), login dengan akun GitHub.
5. Klik **"New app"**.
6. Pilih:
   - Repository: `USERNAME/tpt-jabar-dashboard`
   - Branch: `main`
   - Main file path: `app.py`
7. Klik **"Deploy!"**. Tunggu 1–3 menit sampai proses build & install dependency selesai.
8. Setelah selesai, Anda akan mendapat URL publik seperti:
   `https://tpt-jabar-dashboard-xxxxx.streamlit.app`
9. Setiap kali Anda push perubahan baru ke GitHub (branch `main`), aplikasi akan
   otomatis re-deploy.

## 🔁 Update Data di Masa Depan

Jika ada data tahun baru (misal data aktual 2026 sudah tersedia):
1. Tambahkan baris baru ke `data/data_integrated.csv` mengikuti format kolom yang sama.
2. Push perubahan ke GitHub → aplikasi otomatis ter-update (cache akan refresh
   karena `st.cache_data`/`st.cache_resource` membaca ulang file saat konten berubah,
   atau klik tombol "Rerun" / "Clear cache" di menu titik tiga kanan atas app).

## 🎨 Tema Warna

| Warna | Hex | Penggunaan |
|---|---|---|
| Pink Pastel | `#FFD6E8` / `#FF8FB1` | Aksen, nilai TPT tinggi |
| Ungu Pastel | `#E0C3FC` / `#A66CFF` | Warna utama, sidebar, tombol |
| Biru | `#BDE0FE` / `#4EA8DE` | Aksen, nilai TPT rendah |

## 📊 Fitur Dashboard

- **Beranda** — KPI kabupaten/kota TPT tertinggi & terendah per tahun, ringkasan top 5.
- **Peta Jawa Barat** — choropleth interaktif TPT per kabupaten/kota dengan filter tahun.
- **Tren Provinsi** — grafik tren TPT rata-rata/tertinggi/terendah 2018–2025 + tahun TPT
  provinsi tertinggi/terendah + tren per wilayah.
- **Aktual vs Prediksi** — tabel & visualisasi (bar + scatter) aktual vs prediksi RLB/RF,
  dengan filter skenario (80/20, 70/30, 60/40) dan tahun.
- **Evaluasi Model** — tabel & grafik MAE, RMSE, R² semua model/skenario + feature importance.
- **Prediksi TPT** — input manual faktor (Jumlah Penduduk, TPAK, IPM, PDRB) untuk estimasi
  TPT langsung, serta proyeksi otomatis TPT 2026 seluruh kabupaten/kota (ekstrapolasi tren).
- **Data Lengkap** — tabel data TPT 2018–2025 dengan filter wilayah/tahun + unduh CSV.
