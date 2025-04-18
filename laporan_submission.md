# Laporan Proyek Machine Learning - Hubbal Kholiq Habbaza

## 🧠 Problem Domain: Prediksi Kelulusan Mahasiswa

Pendidikan tinggi merupakan salah satu faktor penentu utama dalam kesuksesan individu di masa depan, baik dari segi ekonomi, sosial, maupun karier. Namun, tidak semua mahasiswa yang masuk perguruan tinggi berhasil menyelesaikan studinya tepat waktu atau bahkan lulus. Tingkat kelulusan mahasiswa menjadi indikator penting dalam mengevaluasi kualitas institusi pendidikan, efektivitas sistem pembelajaran, dan kesejahteraan mahasiswa itu sendiri.

Salah satu tantangan utama dalam dunia pendidikan adalah **mendeteksi secara dini mahasiswa yang berisiko tidak lulus**. Kegagalan mahasiswa dalam menyelesaikan studinya dapat disebabkan oleh berbagai faktor seperti latar belakang sosial ekonomi, kualitas pendidikan sebelumnya, dukungan dari keluarga, serta performa akademik selama masa studi. Oleh karena itu, institusi pendidikan membutuhkan sistem yang dapat **memprediksi kelulusan mahasiswa secara otomatis dan akurat** agar dapat melakukan intervensi yang tepat sebelum terlambat.

Dengan hadirnya teknologi dan data yang lebih terstruktur, **pendekatan berbasis machine learning menjadi solusi potensial** untuk mengatasi permasalahan ini. Dengan menggunakan data akademik dan non-akademik yang tersedia, kita dapat membangun model prediksi yang tidak hanya memperkirakan kemungkinan kelulusan, tetapi juga membantu mengidentifikasi faktor-faktor kunci yang memengaruhinya.

Dalam proyek ini, kami mengembangkan sistem prediksi kelulusan mahasiswa dengan memanfaatkan dataset publik dari Kaggle: *Students Performance in Exams*. Dataset ini berisi informasi mengenai latar belakang siswa seperti gender, status pendidikan orang tua, jenis makan siang, serta nilai ujian matematika, membaca, dan menulis. Melalui model machine learning yang dibangun, kami berharap dapat:

- Memprediksi apakah seorang mahasiswa akan lulus berdasarkan skor ujian dan fitur latar belakang lainnya.
- Memberikan wawasan tentang faktor-faktor yang paling berpengaruh terhadap tingkat kelulusan.
- Menyediakan dasar untuk pengambilan keputusan akademik yang lebih efektif dan berbasis data.

### 🔗 Referensi

- [Students Performance in Exams Dataset - Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- [OECD Education at a Glance 2023](https://www.oecd.org/education/education-at-a-glance/)
- [Predicting Student Academic Performance - ResearchGate](https://www.researchgate.net/publication/328110370_Predicting_Students_Academic_Performance_Using_Data_Mining_Techniques)

## 📊 Business Understanding

### ❓ Problem Statements

1. **Bagaimana cara memprediksi kelulusan mahasiswa berdasarkan nilai ujian dan informasi latar belakangnya?**  
   Banyak institusi kesulitan dalam mengidentifikasi mahasiswa yang berisiko tidak lulus. Jika risiko ini tidak terdeteksi sejak awal, maka institusi gagal memberikan dukungan akademik atau psikologis yang dibutuhkan oleh mahasiswa.

2. **Faktor apa saja yang paling memengaruhi kelulusan mahasiswa?**  
   Selain nilai ujian, ada beberapa faktor sosial dan latar belakang yang kemungkinan besar memengaruhi hasil belajar siswa. Pemahaman terhadap faktor-faktor ini penting untuk membentuk kebijakan dan intervensi pendidikan.

3. **Apakah sistem prediktif dapat membantu lembaga pendidikan dalam merancang strategi intervensi untuk meningkatkan kelulusan?**  
   Dengan mengetahui potensi kegagalan lebih awal, institusi dapat mengambil langkah preventif yang tepat, seperti bimbingan tambahan atau pelatihan keterampilan belajar.

---

### 🎯 Goals

1. **Mengembangkan model machine learning** yang mampu memprediksi status kelulusan mahasiswa berdasarkan data nilai ujian dan fitur latar belakang (seperti jenis kelamin, status pendidikan orang tua, dsb).

2. **Mengidentifikasi fitur-fitur yang paling signifikan** dalam menentukan apakah seorang mahasiswa akan lulus atau tidak. Hal ini akan membantu pengambil kebijakan dalam merancang intervensi berbasis data.

3. **Menyediakan dasar pengambilan keputusan** untuk pihak kampus dalam memberikan perhatian lebih kepada mahasiswa dengan risiko kelulusan rendah, sehingga meningkatkan tingkat kelulusan secara keseluruhan.

---

### 💡 Solution Statements

Untuk meraih tujuan di atas, beberapa pendekatan solusi akan diterapkan:

1. **Baseline Modeling dengan Algoritma Klasifikasi Sederhana**  
   Kami akan memulai dengan algoritma sederhana seperti:
   - Logistic Regression
   - Decision Tree  
   Tujuannya untuk membangun baseline dan memahami struktur data secara umum.

2. **Improvement dengan Model Lanjutan dan Hyperparameter Tuning**  
   Setelah baseline terbentuk, kami akan melakukan tuning dan pengujian model lanjutan seperti:
   - Random Forest Classifier
   - XGBoost Classifier  
   serta menerapkan teknik seperti GridSearchCV untuk mencari parameter terbaik agar akurasi dan generalisasi model meningkat.

3. **Evaluasi Menggunakan Metrik yang Relevan**  
   Kami akan mengevaluasi setiap model menggunakan metrik klasifikasi seperti:
   - Accuracy
   - Precision
   - Recall
   - F1-Score  
   Tujuannya agar solusi yang diambil bisa terukur secara obyektif dan sesuai konteks permasalahan.

---

Dengan pendekatan ini, kami berharap dapat membangun sistem prediksi kelulusan yang efektif dan berguna secara praktis bagi lembaga pendidikan tinggi.

## 📊 Data Understanding

Dataset yang digunakan berjudul **"Students Performance in Exams"**, bersumber dari [Kaggle Datasets](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).

Dataset ini memuat informasi latar belakang siswa serta hasil ujian dalam tiga mata pelajaran: matematika, membaca, dan menulis.

---

### 📐 Ukuran Dataset

- Jumlah baris: **1.000 siswa**
- Jumlah kolom: **8 fitur**
- Tidak terdapat **missing value** pada dataset

---

### 🧾 Penjelasan Setiap Fitur

| Nama Fitur | Tipe | Deskripsi |
|------------|------|-----------|
| `gender` | Kategorikal | Jenis kelamin siswa (`female` atau `male`) |
| `race/ethnicity` | Kategorikal | Kelompok etnis siswa (A, B, C, D, E) |
| `parental level of education` | Kategorikal | Pendidikan tertinggi orang tua (misalnya: `high school`, `bachelor’s degree`, dll.) |
| `lunch` | Kategorikal | Jenis makan siang yang diterima (`standard` atau `free/reduced`) |
| `test preparation course` | Kategorikal | Status keikutsertaan siswa dalam kursus persiapan ujian |
| `math score` | Numerik | Nilai ujian matematika (0–100) |
| `reading score` | Numerik | Nilai ujian membaca (0–100) |
| `writing score` | Numerik | Nilai ujian menulis (0–100) |

---

### 📊 Visualisasi Distribusi Fitur Kategorikal

- Distribusi fitur-fitur kategorikal ditampilkan pada grafik berikut:

#### ✅ Distribusi Gender

![Distribusi Gender](https://github.com/user-attachments/assets/286e810d-7833-47a0-b228-1156fa73559b)

#### ✅ Distribusi Etnis

![Distribusi Etnis](https://github.com/user-attachments/assets/e1ffc957-0a0a-408b-8335-7aead83b1ded)

#### ✅ Distribusi Jenis Makan Siang

![Distribusi Lunch](https://github.com/user-attachments/assets/71f2b9ab-c681-46bf-a9a0-fe431d003ee3)

#### ✅ Distribusi Pendidikan Orang Tua

![Distribusi Pendidikan Orang Tua](https://github.com/user-attachments/assets/e1bc45c0-e59c-46d1-b3d9-bc36117f5f05)

#### ✅ Distribusi Kursus Persiapan Ujian

![Distribusi Kursus Ujian](https://github.com/user-attachments/assets/247c760c-e5bd-4fbd-86a0-08122a7951f2)

---

### 📉 Visualisasi Distribusi Fitur Numerik

- Distribusi nilai siswa pada setiap mata pelajaran terlihat hampir normal.

#### ✅ Distribusi Nilai Matematika

![Distribusi Math Score](https://github.com/user-attachments/assets/218a4c50-e185-4b23-9cf0-83485afff011)

#### ✅ Distribusi Nilai Membaca

![Distribusi Reading Score](https://github.com/user-attachments/assets/c07d5a07-c80d-4cca-b9c6-185701c5a80f)

#### ✅ Distribusi Nilai Menulis

![Distribusi Writing Score](https://github.com/user-attachments/assets/a8304398-f43a-4f00-a00c-6c4d7af99884)

---

### 🔗 Korelasi Antar Skor

- **Reading** dan **Writing Score** memiliki korelasi tinggi (**> 0.9**)
- **Math Score** memiliki korelasi sedang terhadap dua skor lainnya

![Korelasi Skor](https://github.com/user-attachments/assets/42a8c0c8-3226-415f-ab1c-81906a8ba01b)

---

### 📌 Insight Awal

- **Gender**:
  - Perempuan unggul dalam membaca & menulis
  - Laki-laki sedikit unggul di matematika

- **Kursus Persiapan Ujian**:
  - Siswa yang mengikuti kursus cenderung meraih skor lebih tinggi di semua bidang

- **Jenis Makan Siang**:
  - Siswa dengan makan siang standar mencatatkan skor lebih tinggi dibanding yang menerima makan gratis/berkurang

- **Pendidikan Orang Tua**:
  - Terdapat kecenderungan bahwa semakin tinggi pendidikan orang tua, semakin tinggi skor akademik siswa

---

Bagian ini menjadi fondasi awal untuk eksplorasi lebih lanjut sebelum dilakukan *data preprocessing* dan *modeling*.

Dataset yang digunakan berjudul **"Students Performance in Exams"** yang bersumber dari [Kaggle Datasets](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).
Dataset ini memuat informasi mengenai latar belakang siswa serta hasil ujian matematika, membaca, dan menulis.

### 📐 Ukuran Dataset

- Jumlah baris: 1000
- Jumlah kolom: 8
- Tidak terdapat nilai kosong pada dataset.

### 🔍 Tipe Fitur

- **Numerik**:
  - `math score`
  - `reading score`
  - `writing score`
- **Kategorikal**:
  - `gender`
  - `race/ethnicity`
  - `parental level of education`
  - `lunch`
  - `test preparation course`

### 📊 Statistik Deskriptif & Distribusi

- Skor akademik (math, reading, writing) menunjukkan distribusi mendekati normal.
- Ditemukan nilai 0 pada `math score`, yang kemudian diidentifikasi sebagai **outlier**.

### 🔗 Korelasi Antar Skor

- Korelasi tinggi antara `reading score` dan `writing score` (> 0.9).
- Korelasi sedang antara `math score` dan dua skor lainnya.

### 📌 Insight Berdasarkan Kategori

- **Perempuan** unggul di reading dan writing, sedangkan **laki-laki** sedikit unggul di math.
- Siswa yang mengikuti **test preparation course** memperoleh skor yang lebih tinggi secara konsisten.
- **Jenis makan siang** dan **pendidikan orang tua** menunjukkan pengaruh terhadap rata-rata skor siswa.

### 📦 Distribusi Kategorikal

- Distribusi gender cukup seimbang.
- Mayoritas siswa berasal dari etnis grup C dan D.
- Pendidikan orang tua paling umum adalah `some college` dan `associate’s degree`.

Tahapan ini menjadi fondasi penting sebelum dilakukan pembersihan data dan modeling.

## 📊 Data Preparation

Pada tahap ini dilakukan proses pembersihan dan persiapan data sebelum masuk ke tahap pelatihan model machine learning. Berikut beberapa langkah yang dilakukan:

---

#### 1. ✅ Pengecekan & Penanganan Missing Values dan Duplikat

- Dicek apakah terdapat nilai yang hilang (*missing values*) pada setiap kolom menggunakan `df.isnull().sum()`.
- Tidak ditemukan missing values pada dataset.
- Duplikat dicek dan dihapus menggunakan `df.drop_duplicates()` untuk memastikan integritas data.

> 🔍 *Alasan:* Nilai kosong dan data duplikat dapat menyebabkan bias dan memengaruhi hasil pelatihan model.

---

#### 2. 🧹 Deteksi & Penanganan Outlier

- Outlier dicek menggunakan visualisasi boxplot terhadap fitur numerik (`math score`, `reading score`, `writing score`).
- Ditemukan nilai **0** pada `math score` yang dianggap tidak valid sebagai nilai ujian.
- Baris dengan `math score = 0` dihapus dari dataset.

> 🔍 *Alasan:* Nilai ekstrim seperti ini dapat mengganggu distribusi dan menurunkan akurasi model klasifikasi.

---

#### 3. 🏷️ Pembuatan Label Target dan Fitur Tambahan

- Dibuat kolom `average_score` sebagai rata-rata dari ketiga skor akademik.
- Kolom target `pass` ditambahkan:
  - `1` jika `average_score >= 70` (Lulus)
  - `0` jika `average_score < 70` (Tidak Lulus)
- Fitur `score_level` ditambahkan untuk membagi skor ke dalam kategori:
  - **Rendah**: 0–60
  - **Sedang**: 60–80
  - **Tinggi**: 80–100

> 🔍 *Alasan:* Feature engineering ini membantu menyederhanakan proses klasifikasi dan analisis kelulusan.

---

#### 4. ✂️ Split Data: Training dan Testing

- Dataset dibagi menjadi data pelatihan (80%) dan data pengujian (20%) menggunakan `train_test_split`.
- Pembagian dilakukan dengan **stratifikasi berdasarkan label `pass`** untuk menjaga keseimbangan proporsi kelas antara train dan test.

> 🔍 *Alasan:* Stratifikasi mencegah bias distribusi kelas antara data pelatihan dan pengujian.

---

#### 5. 🔠 Encoding Fitur Kategorikal

- Dilakukan encoding pada fitur kategorikal (`gender`, `race/ethnicity`, dll.) menggunakan `LabelEncoder`.
- **Penting:** Encoding hanya dilakukan `.fit()` pada data train, dan `.transform()` digunakan untuk data test untuk menghindari **data leakage**.

> 🔍 *Alasan:* Representasi numerik diperlukan agar algoritma machine learning dapat memproses fitur kategorikal dengan benar.

---

#### 6. 📊 Visualisasi Distribusi Target

- Distribusi label `pass` divisualisasikan setelah proses split menggunakan countplot.
- Ini bertujuan memastikan proporsi kelas seimbang dan tidak terjadi bias setelah pembagian data.

> 🔍 *Alasan:* Visualisasi ini membantu memahami komposisi kelas target yang akan dilatih oleh model.

---

Langkah-langkah ini memastikan bahwa data yang digunakan untuk pelatihan dan pengujian model sudah bersih, terstruktur, dan bebas dari kebocoran data (*data leakage*), sehingga model dapat belajar secara optimal dan memberikan hasil evaluasi yang akurat.

## 🤖 Modeling

Pada tahap ini, dilakukan pembangunan beberapa model machine learning untuk menyelesaikan permasalahan klasifikasi biner: memprediksi kelulusan siswa (`pass = 1` untuk lulus, `0` untuk tidak lulus). Proses modeling terdiri atas dua tahap:

1. **Baseline Modeling** — menggunakan model sederhana tanpa tuning.
2. **Model Improvement (Hyperparameter Tuning)** — pencarian parameter terbaik untuk meningkatkan performa.

---

### 1. ✅ Logistic Regression

#### 📌 Cara Kerja

Logistic Regression adalah model klasifikasi linier yang memetakan input ke probabilitas menggunakan fungsi sigmoid. Kelas ditentukan berdasarkan ambang batas (biasanya 0.5).

#### ⚙️ Parameter

- Menggunakan parameter **default** dari `sklearn.linear_model.LogisticRegression`
- `random_state=42` untuk reproducibility

---

### 2. 🌳 Decision Tree Classifier

#### 📌 Cara Kerja

Decision Tree memisahkan data secara rekursif berdasarkan fitur yang paling mengurangi impurity (Gini/Entropy). Setiap node adalah aturan keputusan, dan leaf node adalah prediksi kelas.

#### ⚙️ Parameter

- Parameter **default** dari `sklearn.tree.DecisionTreeClassifier`
- `random_state=42`

---

## 🔧 Model Improvement (Hyperparameter Tuning)

Setelah baseline model, dilakukan tuning menggunakan **GridSearchCV** untuk meningkatkan akurasi dan generalisasi.

---

### 3. 🌲 Random Forest Classifier (Tuned)

#### 📌 Cara Kerja

Random Forest merupakan ensemble dari banyak pohon keputusan (Decision Tree) yang masing-masing dilatih dengan subset data berbeda, dan digabungkan dengan voting.

#### ⚙️ Parameter Tuning

Grid:

```python
{
    'n_estimators': [100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
```

Best Parameters:

```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

---

### 4. ⚡ XGBoost Classifier (Tuned)

#### 📌 Cara Kerja

XGBoost membangun model secara bertahap, di mana setiap model baru fokus untuk memperbaiki kesalahan dari model sebelumnya, menggunakan teknik boosting berbasis gradien.

#### ⚙️ Parameter Tuning

Grid:

```python
{
    'n_estimators': [100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
```

Best Parameters:

```python
XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    subsample=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
```

---

> 📝 *Catatan:* Hasil evaluasi lengkap disajikan pada section **Evaluation**.

## 📊 Evaluation

Tahap evaluasi ini tidak hanya bertujuan mengukur performa model secara teknis, tetapi juga untuk memastikan bahwa **model yang dibangun benar-benar menjawab kebutuhan bisnis (Business Understanding)** yang telah didefinisikan sebelumnya.

---

### 🔍 Perbandingan Performa Model

Evaluasi dilakukan dengan menggunakan metrik klasifikasi berikut:

- **Accuracy**: Persentase prediksi benar
- **Precision**: Proporsi prediksi positif yang benar
- **Recall**: Proporsi kasus positif yang berhasil dikenali
- **F1-Score**: Harmonik dari precision dan recall
- **Confusion Matrix**: Matriks hasil prediksi aktual vs prediksi

| Model                   | Accuracy | Precision (class 1) | Recall (class 1) | F1-Score (class 1) |
|-------------------------|----------|----------------------|------------------|--------------------|
| Logistic Regression     | 97%      | 0.96                 | 0.98             | 0.97               |
| Decision Tree           | 96%      | 0.95                 | 0.97             | 0.96               |
| Random Forest (Tuned)   | 98%      | 0.99                 | 0.98             | 0.98               |
| XGBoost (Tuned)         | 98%      | 0.99                 | 0.97             | 0.98               |

---

### 📈 Confusion Matrix Model Terbaik (XGBoost)

```
[[107   1]
 [  3  89]]
```

---

### 🎯 Evaluasi Terhadap Business Understanding

#### 1. Apakah model berhasil memprediksi kelulusan mahasiswa secara akurat?

✅ **Ya.**  
Model XGBoost menunjukkan akurasi dan generalisasi yang sangat tinggi (98%), menjadikannya andalan untuk prediksi kelulusan berbasis data.

#### 2. Apakah model mampu mengungkap faktor-faktor yang memengaruhi kelulusan?

✅ **Ya.**  
Fitur yang paling berpengaruh di antaranya: skor `math`, `reading`, `writing`, serta `test preparation course`. Ini sesuai dengan asumsi awal bahwa performa akademik dan dukungan belajar berpengaruh signifikan.

#### 3. Apakah solusi prediktif ini dapat digunakan untuk merancang intervensi yang efektif?

✅ **Ya.**  
Institusi dapat memanfaatkan hasil prediksi untuk:

- Mengidentifikasi siswa yang berisiko tidak lulus
- Memberikan mentoring atau remedial
- Mengalokasikan sumber daya bimbingan belajar

---

### 🧠 Dampak Bisnis dan Manfaat Praktis

- 🎯 **Intervensi Dini:** Sistem prediksi dapat digunakan sebagai alat bantu untuk menyaring siswa yang membutuhkan perhatian lebih sejak awal semester.
- 🎯 **Efektivitas Program:** Terbukti bahwa kursus persiapan ujian berdampak positif pada skor siswa.
- 🎯 **Pengambilan Keputusan Berbasis Data:** Model mendukung pendekatan berbasis data dalam perencanaan akademik dan pengelolaan risiko kelulusan.

---

### ✅ Kesimpulan

- Seluruh model menunjukkan performa baik, namun **XGBoost menjadi model terbaik secara teknis dan bisnis**.
- Evaluasi menunjukkan bahwa semua **problem statement dan business goals berhasil dicapai**.
- Solusi yang dirancang terbukti **berdampak nyata dalam konteks pendidikan** dan dapat langsung diterapkan di lapangan.

Dengan demikian, sistem ini memiliki potensi besar untuk diimplementasikan sebagai bagian dari *Decision Support System* yang strategis di lingkungan perguruan tinggi.
