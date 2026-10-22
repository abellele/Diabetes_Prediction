# 🚀 Cara Menjalankan Aplikasi SugarSense

## Pilihan 1: Menjalankan Aplikasi dengan UI/UX Lengkap (Recommended)

Aplikasi ini memiliki tampilan yang indah dengan multi-page design sesuai dengan desain UI/UX yang telah dibuat.

```bash
streamlit run app.py
```

**Fitur:**
- ✅ Home/About Page dengan informasi lengkap
- ✅ Input Page dengan form yang user-friendly
- ✅ Result Pages dengan visualisasi chart yang menarik
- ✅ Responsive design
- ✅ Smooth transitions dan animations

## Pilihan 2: Menjalankan Aplikasi Sederhana

Jika Anda ingin versi yang lebih sederhana tanpa halaman About:

```bash
streamlit run diabetes_predict.py
```

**Fitur:**
- ✅ Simple input form
- ✅ Quick prediction results
- ✅ Basic UI

## 📋 Langkah-Langkah Detail

### 1. Pastikan Environment Sudah Siap

Buka terminal/command prompt dan navigasi ke folder ini:

```bash
cd "c:\Users\tjoen\OneDrive\Desktop\THESIS CODE\CODE\Diabetes_Prediction\STREAMLIT_SKRIPSI"
```

### 2. Install Dependencies (Jika Belum)

```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi

**Untuk UI lengkap:**
```bash
streamlit run app.py
```

**Untuk UI sederhana:**
```bash
streamlit run diabetes_predict.py
```

### 4. Akses Aplikasi

Setelah command dijalankan, aplikasi akan otomatis membuka di browser pada alamat:
```
http://localhost:8501
```

Jika tidak otomatis terbuka, copy paste URL tersebut ke browser Anda.

## 🎯 Tips Penggunaan

### Pada Home Page (app.py):
1. Baca informasi tentang SugarSense
2. Klik tombol "🚀 Start Prediction" untuk mulai

### Pada Input Page:
1. Masukkan data kesehatan Anda
2. Gunakan nilai default yang sudah disediakan sebagai referensi
3. Klik "🔬 Get Prediction!" untuk melihat hasil

### Pada Result Page:
1. Lihat hasil prediksi (Positive/Negative)
2. Perhatikan chart probability
3. Baca rekomendasi yang diberikan
4. Klik "🔄 New Prediction" untuk prediksi baru

## 📊 Contoh Data untuk Testing

Form prediksi memiliki 14 input.

**Data Normal (Non-Diabetes):**
- Age: 30 tahun
- Gender: Female
- Height: 160 cm
- Weight: 65 kg
- BMI: 22.5
- Glucose: 5.5 mmol/L
- Pulse Rate: 75 bpm
- Systolic BP: 120 mmHg
- Diastolic BP: 80 mmHg
- Hypertensive: No (0)
- Family Hypertension: No (0)
- Family Diabetes: No (0)
- Cardiovascular Disease: No (0)
- Stroke: No (0)

**Data High Risk (Diabetes):**
- Age: 60 tahun
- Gender: Male
- Height: 165 cm
- Weight: 85 kg
- BMI: 32.0
- Glucose: 15.0 mmol/L
- Pulse Rate: 95 bpm
- Systolic BP: 160 mmHg
- Diastolic BP: 100 mmHg
- Hypertensive: Yes (1)
- Family Hypertension: Yes (1)
- Family Diabetes: Yes (1)
- Cardiovascular Disease: No (0)
- Stroke: No (0)

## 🛑 Troubleshooting

### Error: "Model not found"
- Pastikan file `model.pkl` ada di folder yang sama dengan `app.py`

### Error: "Module not found"
- Jalankan: `pip install -r requirements.txt`

### Port sudah digunakan
- Tambahkan parameter port berbeda:
  ```bash
  streamlit run app.py --server.port 8502
  ```

### Aplikasi tidak terbuka otomatis
- Buka browser manual dan akses: `http://localhost:8501`

## 📱 Akses dari Device Lain

Jika ingin mengakses dari HP/device lain di jaringan yang sama:

1. Cari IP address komputer Anda:
   ```bash
   ipconfig
   ```
   (Cari "IPv4 Address")

2. Jalankan aplikasi dengan:
   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```

3. Akses dari device lain:
   ```
   http://[IP_ADDRESS]:8501
   ```
   Contoh: `http://192.168.1.100:8501`

## 🎨 Perbedaan Antara app.py dan diabetes_predict.py

| Fitur | app.py | diabetes_predict.py |
|-------|--------|---------------------|
| Multi-page | ✅ Yes | ❌ No |
| About Page | ✅ Yes | ❌ No |
| Visual Charts | ✅ Plotly | ❌ No |
| Custom Styling | ✅ Extensive | ⚠️ Minimal |
| Navigation | ✅ Yes | ❌ No |
| Responsive | ✅ Yes | ⚠️ Basic |
| Animations | ✅ Yes | ❌ No |

## ⚠️ Important Notes

- Pastikan Python 3.8+ terinstall
- File `model.pkl` harus ada di folder yang sama
- Untuk production deployment, gunakan platform seperti Streamlit Cloud, Heroku, atau AWS

## 🌐 Deploy ke Streamlit Cloud

1. Push semua file ke GitHub repository
2. Buka [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect repository Anda
4. Pilih file `app.py` sebagai main file
5. Deploy!

---

**Selamat mencoba! 🎉**

Jika ada pertanyaan atau error, pastikan semua dependencies sudah terinstall dengan benar.
