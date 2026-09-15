# **HOTEL BOOKING CANCELLATION PREDICTION**

## 1. Project Overview

### Problem Statement

Kamar hotel merupakan *perishable inventory*, dimana ketika satu malam terlewat tanpa terjual, potensi pendapatan dari kamar tersebut hilang secara permanen, sementara biaya operasional seperti staf, *housekeeping*, dan listrik tetap berjalan. Saat ini, hotel belum memiliki mekanisme sistematis untuk mengidentifikasi reservasi yang memiliki risiko *cancellation* tinggi. Akibatnya, tindakan preventif cenderung bersifat reaktif dan baru dilakukan mendekati tanggal check-in, ketika waktu yang tersedia untuk menjual kembali kamar yang dibatalkan sudah sangat terbatas.

Kondisi tersebut dapat menimbulkan beberapa dampak bisnis, yaitu *revenue loss* akibat kamar yang dibatalkan dan tidak sempat dijual kembali, *occupancy forecasting* yang kurang akurat, perencanaan operasional seperti *housekeeping*, F&B, dan front office yang kurang optimal, serta keputusan *overbooking* yang belum didukung oleh estimasi risiko *cancellation* pada tingkat reservasi.

Proyek ini berfokus pada prediksi kemungkinan pembatalan reservasi (`is_canceled`) menggunakan data historis booking dari hotel di Portugal, yaitu *City Hotel* di Lisbon dan *Resort Hotel* di Algarve. Model dirancang untuk memberikan estimasi risiko *cancellation* pada setiap reservasi sehingga hotel dapat beralih dari pendekatan reaktif menjadi lebih proaktif, dengan menyesuaikan tindakan berdasarkan tingkat risiko masing-masing reservasi daripada menerapkan perlakuan yang sama terhadap seluruh reservasi.

### Key Objectives

* Mengidentifikasi reservasi berisiko tinggi dibatalkan sejak awal, sehingga tim reservasi dapat melakukan tindakan preventif yang lebih terarah.
* Meningkatkan akurasi *forecast* okupansi, sehingga perencanaan *housekeeping*, sta, dan strategi harga menjadi lebih presisi.
* Mendukung strategi *overbooking* yang lebih terukur berdasarkan probabilitas pembatalan per segmen.
* Menurunkan potensi *revenue loss* akibat kamar yang batal terpakai namun tidak sempat dijual kembali.

---

## 2. Data Sources

Dataset ini bersumber dari *Kaggle* yang dapat diakses di [kaggle.com/hotel_demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data) dimana dataset ini mencakup data *booking* hotel yang diambil di Portugal dengan 2 jenis hotel, yaitu *City Hotel* di Lisbon dan *Resort Hotel* di Algarve.

### Dataset

* **File**: `hotel_bookings_2017.csv`
* **Jumlah baris awal:** 199.390 *booking*
* **Jumlah kolom awal:** 32 fitur
* **Unit analisis:** 1 baris mewakili 1 catatan *booking* hotel
* **Target:** `is_canceled` (`1`= dibatalkan, `2`= tidak dibatalkan)
* **Periode data:** 1 Juli 2015 - 31 Agustus 2017

---

## 3. Technologies Used

### Programming Language

* Python

### Data Manipulation & Analysis

* Pandas
* Numpy

### Data Visualization

* Matplotlib
* Seaborn

### Statistical Analysis

* Scipy
* Statsmodels

### Machine Learning

* Classification algorithms
* Hyperparameter tuning
* Model evaluation
* Scikit-learn
* XGBoost
* Imblearn
* Category_encoders
* SHAP
* Joblib

### Development Environment

* Jupyter Notebook
* Visual Studio Code

### Version Control

* Git
* GitHub

---

## 4. Project Structure

```text
├── data/
│   ├── split/
│   │   ├── test.csv                                                <- data uji hasil splitting temporal
│   │   ├── train.csv                                               <- data latih hasil splitting temporal
│   │   └── validation.csv                                          <- data validasi hasil splitting temporal
│   ├── hotel_booking_2017_cleaned.csv                              <- dataset final setelah cleaning
│   └── hotel_bookings_2017.csv                                     <- dataset mentah awal
├── final_model/
│   └── XGBoost.pkl                                                 <- model final
├── notebooks/                                                      <- jupyter notebook yang mencakup alur pengerjaan
├── reports/                                                        <- hasil interpretasi dan implementasi model
├── src/
│   └── app.py                                                      <- streamlit
└── README.md
```

---

## 5. Analysis & Machine Learning Workflow

### 5.1 Data Understanding

Audit menyeluruh terhadap `hotel_bookings_2017.csv` mencakup pemeriksaan struktur & tipe data, *missing value*, *duplicate records*, distribusi variabel numerik/kategorikal, serta distribusi target.

## 5.2 Data Cleaning & Feature Engineering

*Data cleaning* meliputi langkah-langkah berikut ini:

* Penghapusan fitur yang berpotensi mengakibatkan *leakage*.
* Standarisasi kategori `Undefined` menjadi `Unknown`.
* Penanganan *missing values*.
* Penanganan *outlier*.
* Menambahkan fitur turunan yang mungkin berguna untuk analisis dan penyusunan model.

### 5.3 Exploratory Data Analysis

Eksplorasi data analisis dilakukan untuk:

* Memahami distribusi status pembatalan reservasi (`is_canceled`).
* Menelaah karakteristik dan pola perilaku reservasi.
* Mengidentifikasi hubungan antara fitur dengan status pembatalan.
* Mendeteksi pola yang berpotensi menunjukkan peningkatan risiko *cancellation*.

### 5.4 Machine Learning Modeling

Tahap *modeling* dilakukan untuk membandingkan performa awal beberapa algoritma klasifikasi dalam memprediksi reservasi yang berpotensi *cancel* dan menyaring kandidat yang layak dilanjutkan, mencari kombinasi *preprocessing* dan *hyperparameter* terbaik untuk tiap kandidat model, memilih model final berdasarkan skor akurasi, serta menentukan *threshold* klasifikasi yang sesuai dengan kebutuhan bisnis.

### 5.5 Model Evaluation

Karena kamar hotel bersifat *perishable inventory*, hotel perlu membedakan reservasi berisiko *cancel* secara andal sekaligus menjaga proporsi prediksi yang benar secara keseluruhan, sehingga model dievaluasi menggunakan ROC-AUC dan akurasi.

* **ROC-AUC** mengukur kemampuan model membedakan reservasi yang akan *cancel* dan tidak *cancel* pada berbagai kemungkinan *threshold*, sehingga digunakan sebagai metrik utama untuk *model comparison*, terutama karena tetap relevan pada kondisi *moderate class imbalance* seperti pada dataset ini.
* **Akurasi** mengukur proporsi keseluruhan prediksi yang benar pada kedua kelas, dan digunakan sebagai dasar penentuan *threshold* optimal agar hotel memperoleh titik keseimbangan yang wajar antara risiko *revenue loss* (*False Negative*) dan risiko *walk situation*/biaya intervensi berlebihan (*False Positive*).

---

## 6. Summary of Findings

### 6.1 Business Insights

Analisis ini memberikan gambaran tentang pola reservasi dan faktor-faktor yang berkaitan dengan *cancellation*.

Pendekatan *machine learning* memungkinkan hotel bergeser dari strategi retensi/preventif yang reaktif menuju strategi yang lebih proaktif dan berbasis data, dengan mengidentifikasi reservasi yang berisiko tinggi dibatalkan sejak awal, bukan menunggu mendekati tanggal *check-in*.

Temuan dari analisis ini meliput:

* Risiko *cancellation* dapat diprediksi menggunakan kombinasi karakteristik reservasi, riwayat *booking*, dan sumber pemesanan.
* Beberapa fitur secara konsisten berkontribusi terhadap prediksi tanpa indikasi *data leakage*.
* Model dapat membantu memprioritaskan reservasi yang memerlukan perhatian tim reservasi, alih-alih memperlakukan seluruh *booking* dengan cara yang sama.

### 6.2 Actionable Recommendations

Berdasarkan hasil analisis dan prediksi *cancellation*, hotel disarankan untuk:

1. **Prioritaskan Reservasi Berisiko Tinggi**
   Gunakan model prediksi *cancellation* untuk mengidentifikasi reservasi dengan probabilitas *cancel* tinggi dan prioritaskan untuk tindakan preventif.
2. **Terapkan Tindakan Preventif Secara Proaktif**
   Hubungi atau konfirmasi ulang reservasi berisiko tinggi sebelum mendekati tanggal kedatangan, alih-alih menunggu setelah reservasi benar-benar dibatalkan.
3. **Sesuaikan Tindakan dengan Tingkat Risiko**
   Terapkan pendekatan bertingkat, dimana *booking* dengan *Low Risk* cukup dipantau rutin, *Medium Risk* menerima *reminder* otomatis, dan *High Risk* menerima aksi administratif langsung (*reconfirmation*).
4. **Audit Fitur Berpotensi *Data Leakage* Sebelum *Deployment Penuh***
   Pastikan status ketersediaan `deposit_type` dan `room_type_changed` pada saat prediksi benar-benar dibutuhkan (sebelum kedatangan tamu). Jika tidak tersedia, model perlu dilatih ulang tanpa fitur ini.
5. **Pantau dan Latih Ulang Model Secara Berkala**
   Perbarui model secara berkala mengingat adanya indikasi *concept drift*, serta pantau performa berdasarkan hotel, segmen, *channel*, dan musim.
6. **Ukur Efektivitas Intervensi**
   Validasi asumsi *cost-benefit* melalui pilot terbatas sebelum scale-up, dengan memantau perubahan *cancellation rate* dan *revenue* yang berhasil dipertahankan.

---

## 7. Business Impact

Dampak bisnis yang diharapkan dari proyek ini adalah membantu hotel untuk:

* Meningkatkan ketepatan *occupancy forecasting*.
* Menurunkan potensi *revenue loss* akibat kamar yang batal terpakai namun tidak sempat dijual kembali.
* Meningkatkan efisiensi tindakan preventif dengan memprioritaskan reservasi berisiko tinggi.
* Mendukung alokasi sumber daya tim reservasi secara lebih terarah, bukan menyeluruh ke semua *booking*.
* Mendukung strategi *overbooking* yang lebih terukur berdasarkan probabilitas cancellation per segmen.
* Memberi waktu lebih awal bagi tim reservasi untuk melakukan tindakan preventif.

---

## 8. Contact

* **Name:** Zian Carlos Wong, Nadya Divia Go, Angela Adytha Putri
* **Email:** ziancrlswong@gmail.com, nadya.diviago1612@gmail.com, angel.adytha@gmail.com
* **GitHub:** [ziancarlos](https://github.com/ziancarlos), [nadya1612](https://github.com/nadya1612), [watermallencute](https://github.com/watermallencute)
