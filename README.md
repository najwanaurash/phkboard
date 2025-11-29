# Dashboard Analisis PHK Indonesia

## 🎯 Deskripsi

Dashboard interaktif untuk menganalisis tren pemutusan hubungan kerja (PHK) di Indonesia. Dashboard ini menyediakan visualisasi yang informatif dan elegant dengan berbagai fitur analisis menggunakan teknik Clustering dan Regresi Linear.

## ✨ Fitur Utama

### 1. 📊 Visualisasi Interaktif
- **Trend Analysis** - Tren PHK per tahun dengan area chart
- **Distribution Analysis** - Distribusi skala PHK (Kecil, Sedang, Besar)
- **Top Rankings** - Top 10 provinsi dan sektor dengan PHK tertinggi
- **Heatmap** - Visualisasi interaktif provinsi vs tahun
- **Correlation Matrix** - Matriks korelasi antar variabel

### 2. 🔍 Clustering Analysis
- **K-Means Clustering** - Pengelompokan provinsi berdasarkan pola PHK
- **PCA Visualization** - Visualisasi 2D hasil clustering
- **Feature Importance** - Analisis faktor paling berpengaruh
- **Cluster Characteristics** - Karakteristik setiap cluster

### 3. 📈 Regression Analysis
- **Multiple Linear Regression** - Prediksi PHK berdasarkan multiple features
- **Model Evaluation** - R² Score, RMSE, MAE metrics
- **Feature Coefficients** - Analisis pengaruh setiap variabel
- **Prediction Tool** - Tool interaktif untuk prediksi PHK

### 4. 🎨 User Interface
- **Mode Gelap & Terang** (Dark/Light Mode Toggle)
- Sidebar dengan multiple filters:
  - Filter Tahun
  - Filter Provinsi
  - Filter Sektor
  - Filter Skala PHK
- Responsive design dengan layout wide
- Interactive charts dengan Plotly

### 5. 📥 Export Functionality
- Download data filtered dalam format CSV
- Preview data yang difilter
- Statistik deskriptif

## 🚀 Cara Menjalankan

### Instalasi Dependencies

```bash
pip install -r requirements.txt
