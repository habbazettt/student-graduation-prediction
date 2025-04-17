# -*- coding: utf-8 -*-
"""Submission_MLT_Hubbal.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1acvm3xQK3XZC9To65jLLPJoR9pU5d_Ry

# **Import Libraries**

Pada tahap ini dilakukan import seluruh library yang dibutuhkan untuk eksplorasi data, visualisasi, pemodelan, hingga evaluasi model. Beberapa library penting antara lain `pandas` untuk manipulasi data, `matplotlib` dan `seaborn` untuk visualisasi, serta `sklearn` untuk modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Atur style visualisasi
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

"""# **Data Loading**

Dataset yang digunakan merupakan data performa siswa yang bersumber dari Kaggle. Dataset dimuat ke dalam dataframe menggunakan `pandas.read_csv()`.

`df.head()` digunakan untuk melihat struktur awal data, termasuk fitur-fitur yang tersedia serta bentuk nilai pada setiap kolom.
"""

df = pd.read_csv('/content/StudentsPerformance.csv')

# Tampilkan 5 baris awal
print("Preview Data:")
display(df.head())

"""# **Data Understanding**

Pada tahap ini, dilakukan proses pemahaman awal terhadap dataset yang digunakan. Tujuan utamanya adalah untuk memperoleh gambaran umum mengenai struktur data, tipe data, jumlah nilai kosong (missing values), dan karakteristik statistik dasar. Tahapan ini sangat penting sebelum melanjutkan ke proses pembersihan dan eksplorasi data lebih lanjut.

---

### 🔹 Ukuran Dataset
Kode:
```python
print(df.shape)
```
Menampilkan ukuran dataset dalam format `(jumlah baris, jumlah kolom)`. Informasi ini berguna untuk mengetahui seberapa besar cakupan data yang akan dianalisis.

---

### 🔹 Informasi Tipe Data
Kode:
```python
print(df.info())
```
Menampilkan tipe data dari setiap kolom serta jumlah data non-null. Ini membantu mengidentifikasi apakah kolom bertipe numerik, kategorikal, atau object (teks), dan apakah terdapat kemungkinan konversi tipe data yang perlu dilakukan.

---

### 🔹 Statistik Deskriptif
Kode:
```python
display(df.describe())
```
Menampilkan ringkasan statistik deskriptif untuk kolom numerik, seperti nilai minimum, maksimum, rata-rata (mean), standar deviasi, dan kuartil. Dari sini bisa diketahui distribusi awal data dan potensi keberadaan outlier.

---

### 🔹 Cek Missing Values
Kode:
```python
print(df.isnull().sum())
```
Digunakan untuk menghitung jumlah nilai kosong (missing values) di setiap kolom. Jika terdapat nilai kosong, perlu dipertimbangkan metode penanganan seperti:
- Mengisi nilai kosong (imputasi) dengan mean/median/modus
- Menghapus baris atau kolom yang memiliki terlalu banyak missing value

---

### 🔹 Nilai Unik per Kolom
Kode:
```python
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")
```
Menghitung jumlah nilai unik pada setiap kolom. Ini berguna untuk:
- Mengidentifikasi apakah kolom merupakan data kategorikal
- Mengenali kolom dengan nilai konstan atau terlalu bervariasi (misalnya identifier)

---

### ✨ Insight Awal:
- Dataset memiliki struktur dan ukuran tertentu yang bisa mempengaruhi pendekatan analisis selanjutnya.
- Terdapat kolom-kolom dengan nilai kosong yang perlu ditangani agar tidak mengganggu kualitas analisis.
- Statistik deskriptif membantu memahami karakteristik dasar data numerik, seperti kecenderungan distribusi dan pencilan (outlier).
- Jumlah nilai unik per kolom membantu klasifikasi awal tipe data (numerik vs kategorikal) dan fitur mana yang relevan untuk model nantinya.

Tahap ini sangat penting sebagai dasar sebelum masuk ke proses *Data Cleaning* dan *Exploratory Data Analysis (EDA)*.
"""

print("\nUkuran Dataset:")
print(df.shape)

print("\nInformasi Tipe Data:")
print(df.info())

print("\nStatistik Deskriptif:")
display(df.describe())

print("\nCek Missing Values:")
print(df.isnull().sum())

print("\nNilai Unik per Kolom:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")

"""# **Exploratory Data Analysis (EDA)**

Pada tahap ini, dilakukan eksplorasi awal terhadap dataset untuk memahami distribusi data, hubungan antar fitur, serta potensi pola-pola yang dapat bermanfaat dalam tahap pemodelan.

### 1. Distribusi Fitur Numerik
Distribusi tiga fitur numerik utama, yaitu **math score**, **reading score**, dan **writing score**, divisualisasikan dalam bentuk histogram dengan KDE (Kernel Density Estimation). Visualisasi ini membantu melihat sebaran nilai dari masing-masing skor.

**Insight:**
- Sebagian besar nilai berada pada rentang menengah hingga tinggi.
- Terdapat kecenderungan distribusi mendekati normal, meskipun tidak sepenuhnya simetris.

### 2. Korelasi Antar Skor Akademik
Dilakukan analisis korelasi antar ketiga fitur numerik tersebut dengan menggunakan **heatmap**. Korelasi ini memberikan gambaran apakah skor satu mata pelajaran berkaitan dengan skor lainnya.

**Insight:**
- Terdapat korelasi positif yang cukup kuat antar ketiga skor.
- Ini mengindikasikan bahwa siswa yang memiliki nilai tinggi di satu bidang cenderung memiliki nilai tinggi di bidang lainnya juga.

### 3. Penambahan Fitur `average_score`
Untuk mempermudah analisis dan visualisasi, ditambahkan fitur baru bernama `average_score` yang merupakan rata-rata dari ketiga skor akademik. Fitur ini digunakan dalam beberapa visualisasi berikutnya untuk melihat tren secara keseluruhan.

### 4. Rata-rata Skor Berdasarkan Fitur Kategorikal
Dianalisis perbandingan skor rata-rata berdasarkan kategori:
- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch
- Test Preparation Course

Visualisasi menggunakan boxplot untuk melihat distribusi dan perbedaan skor rata-rata antar kategori.

**Insight:**
- Siswa yang mengikuti program persiapan ujian umumnya memiliki nilai rata-rata lebih tinggi.
- Terdapat variasi skor berdasarkan kategori lainnya seperti jenis kelamin dan tingkat pendidikan orang tua.

### 5. Distribusi Data Kategorikal
Untuk fitur kategorikal, dilakukan visualisasi menggunakan countplot untuk melihat jumlah siswa dalam setiap kategori.

**Insight:**
- Distribusi kategori relatif seimbang, meskipun beberapa kategori seperti jenis ras atau tingkat pendidikan orang tua menunjukkan perbedaan jumlah yang cukup signifikan.

---

Analisis ini memberikan gambaran umum tentang struktur data serta membantu dalam pengambilan keputusan pada tahap preprocessing dan pemodelan berikutnya.
"""

# Fitur numerik dan kategorikal
num_features = ['math score', 'reading score', 'writing score']
cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

# Distribusi Skor Numerik
for feature in num_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribusi {feature.title()}')
    plt.xlabel(feature.title())
    plt.ylabel('Jumlah Siswa')
    plt.grid(True)
    plt.show()

# Korelasi Antar Skor
plt.figure(figsize=(6, 4))
sns.heatmap(df[num_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("📊 Korelasi Antar Skor Akademik")
plt.show()

# Tambahkan kolom rata-rata skor untuk analisis lanjutan
df['average_score'] = df[num_features].mean(axis=1)

# Perbandingan Rata-rata Skor Berdasarkan Fitur Kategorikal
for feature in cat_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=feature, y='average_score', palette='pastel')
    plt.title(f'📌 Rata-rata Skor Berdasarkan {feature.title()}')
    plt.xticks(rotation=45)
    plt.xlabel(feature.title())
    plt.ylabel('Rata-rata Skor')
    plt.grid(True)
    plt.show()

# Countplot Kategori (Distribusi Data Kategorikal)
for feature in cat_features:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=feature, palette='muted')
    plt.title(f'Jumlah Siswa per {feature.title()}')
    plt.xticks(rotation=30)
    plt.ylabel('Jumlah')
    plt.grid(axis='y')
    plt.show()

"""# **Data Preprocessing**

Pada tahap ini dilakukan proses pembersihan dan persiapan data sebelum masuk ke tahap pelatihan model machine learning. Berikut beberapa langkah yang dilakukan:

---

#### 1. ✅ Pengecekan & Penanganan Missing Values dan Duplikat
- Dicek apakah terdapat nilai yang hilang (*missing values*) pada setiap kolom.
- Ditemukan bahwa tidak ada missing values.
- Data duplikat juga dihapus untuk memastikan kualitas data yang lebih baik.

#### 2. 🧹 Deteksi & Penanganan Outlier
- Dilakukan visualisasi menggunakan boxplot untuk mendeteksi outlier pada skor-skor ujian.
- Ditemukan bahwa terdapat nilai `0` pada kolom `math score` yang dianggap tidak realistis.
- Baris dengan nilai tersebut dihapus dari dataset.

#### 3. 🏷️ Pembuatan Label Target dan Fitur Tambahan
- Dibuat kolom baru `average_score` sebagai rata-rata dari tiga nilai ujian: matematika, membaca, dan menulis.
- Berdasarkan `average_score`, dibuat label target `pass`:
  - `1` jika rata-rata skor ≥ 70 (lulus)
  - `0` jika < 70 (tidak lulus)
- Juga dibuat fitur kategorikal `score_level` untuk membagi skor ke dalam kategori: Rendah, Sedang, dan Tinggi.

#### 4. ✂️ Split Data: Training dan Testing
- Dataset dibagi menjadi data pelatihan (80%) dan data pengujian (20%).
- Split dilakukan dengan **stratifikasi berdasarkan label `pass`** agar proporsi kelas tetap seimbang antara train dan test.

#### 5. 🔠 Encoding Fitur Kategorikal
- Dilakukan proses encoding menggunakan `LabelEncoder` untuk mengubah nilai kategorikal menjadi numerik.
- Penting: Encoding hanya di-*fit* pada data training dan kemudian di-*transform* ke data test untuk mencegah data leakage.

#### 6. 📊 Visualisasi Distribusi Target
- Dilakukan visualisasi distribusi label target pada data training untuk memastikan stratifikasi berhasil.

---

Langkah-langkah ini memastikan bahwa data yang digunakan untuk pelatihan dan pengujian model sudah bersih, terstruktur, dan bebas dari kebocoran data (*data leakage*), sehingga model dapat belajar secara optimal dan memberikan hasil evaluasi yang akurat.

## **Pembersihan Awal**
"""

# Cek & Tangani Missing Values
print("Jumlah Missing Values per Kolom:")
print(df.isnull().sum())

# Hapus Data Duplikat
print(f"Jumlah data sebelum hapus duplikat: {df.shape}")
df.drop_duplicates(inplace=True)
print(f"Jumlah data setelah hapus duplikat: {df.shape}")

# Deteksi dan tangani outlier (math score = 0 dianggap tidak realistis)
df = df[df['math score'] > 0]

# Buat average_score dan label target pass
num_features = ['math score', 'reading score', 'writing score']
df['average_score'] = df[num_features].mean(axis=1)
df['pass'] = df['average_score'].apply(lambda x: 1 if x >= 70 else 0)

# Binning skor ke dalam kategori
df['score_level'] = pd.cut(df['average_score'],
                           bins=[0, 60, 80, 100],
                           labels=['Rendah', 'Sedang', 'Tinggi'])

"""## **Split Data**"""

X = df.drop(columns=['average_score', 'pass', 'score_level'])  # fitur
y = df['pass']  # target

# Split dengan stratifikasi agar proporsi kelas seimbang
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

"""## **Encoding Fitur Kategorikal**"""

cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
le = LabelEncoder()

# Terapkan LabelEncoder hanya di training, lalu transform test
for col in cat_features:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

"""## **Visualisasi target setelah split**"""

sns.countplot(x=y_train)
plt.title("Distribusi Target di Training Set")
plt.xlabel("Kelulusan")
plt.ylabel("Jumlah Siswa")
plt.show()

"""# **Model Building**

Pada tahap ini, dilakukan pelatihan model dasar (*baseline*) sebagai langkah awal untuk membandingkan performa model yang lebih kompleks di tahap selanjutnya. Dua model dasar yang digunakan adalah:

1. **Logistic Regression**
2. **Decision Tree Classifier**

---

### 🔧 Langkah-langkah:

1. **Inisialisasi Model**
   - Logistic Regression dan Decision Tree diinisialisasi dengan `random_state=42` agar hasil eksperimen dapat direproduksi.

2. **Pelatihan Model**
   - Kedua model dilatih menggunakan data training (`X_train`, `y_train`).

3. **Prediksi**
   - Setelah pelatihan, model digunakan untuk memprediksi label pada data uji (`X_test`).

4. **Evaluasi Model**
   - Performa model dievaluasi menggunakan:
     - `classification_report`: Menyediakan metrik **precision**, **recall**, **f1-score**, dan **accuracy**.
     - `confusion_matrix`: Menampilkan jumlah prediksi benar dan salah untuk masing-masing kelas (Lulus / Tidak Lulus).

---

### 📈 Hasil Evaluasi:

#### ✅ Logistic Regression
- **Akurasi:** 97%
- **Classification Report:**

  ```
                precision    recall  f1-score   support

             0       0.98      0.96      0.97       108
             1       0.96      0.98      0.97        92

      accuracy                           0.97       200
     macro avg       0.97      0.97      0.97       200
  weighted avg       0.97      0.97      0.97       200
  ```

- **Confusion Matrix:**

  ```
  [[104   4]
   [  2  90]]
  ```

- 📌 **Insight:**
  - Precision & recall untuk kedua kelas sangat tinggi.
  - Hanya 6 kesalahan prediksi dari total 200 data uji:
    - 4 siswa tidak lulus diprediksi lulus
    - 2 siswa lulus diprediksi tidak lulus

---

#### 🌳 Decision Tree Classifier
- **Akurasi:** 96%
- **Classification Report:**

  ```
                precision    recall  f1-score   support

             0       0.97      0.95      0.96       108
             1       0.95      0.97      0.96        92

      accuracy                           0.96       200
     macro avg       0.96      0.96      0.96       200
  weighted avg       0.96      0.96      0.96       200
  ```

- **Confusion Matrix:**

  ```
  [[103   5]
   [  3  89]]
  ```

- 📌 **Insight:**
  - Performa model sangat baik dan sebanding dengan Logistic Regression.
  - Terdapat sedikit lebih banyak kesalahan prediksi dibanding Logistic Regression, namun masih tergolong sangat kecil.

---

### ✅ Kesimpulan:
Model-model ini berfungsi sebagai **baseline**, yang akan menjadi pembanding terhadap model lain yang akan digunakan selanjutnya, seperti model yang dioptimasi dengan **hyperparameter tuning** atau **ensemble methods**.
"""

# Inisialisasi model
logreg = LogisticRegression(random_state=42)
tree = DecisionTreeClassifier(random_state=42)

# Training model
logreg.fit(X_train, y_train)
tree.fit(X_train, y_train)

# Prediksi
y_pred_logreg = logreg.predict(X_test)
y_pred_tree = tree.predict(X_test)

# Evaluasi Logistic Regression
print("=== Logistic Regression Report ===")
print(classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logreg))

# Evaluasi Decision Tree
print("\n=== Decision Tree Classifier Report ===")
print(classification_report(y_test, y_pred_tree))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

"""# **Evaluation dan Model Improvement (Hyperparameter Tuning)**

Setelah membangun baseline model, tahap selanjutnya adalah meningkatkan performa model dengan melakukan **hyperparameter tuning** menggunakan teknik **GridSearchCV**. Dua model yang dituning adalah:

1. **Random Forest Classifier**
2. **XGBoost Classifier**

### Evaluasi:
#### Random Forest (Tuned)
- Akurasi: **98%**
- Sangat sedikit kesalahan prediksi.
- Confusion Matrix:
  ```
  [[107   1]
   [  2  90]]
  ```

#### XGBoost (Tuned)
- Akurasi: **98%**
- Sama baiknya dengan Random Forest.
- Confusion Matrix:
  ```
  [[107   1]
   [  3  89]]
  ```

### Kesimpulan:
- Kedua model setelah tuning menunjukkan performa sangat baik.
- Akan dipertimbangkan untuk memilih model akhir berdasarkan kecepatan pelatihan, kompleksitas, dan interpretabilitas.
"""

# GridSearch untuk Random Forest
rf_param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_clf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(estimator=rf_clf, param_grid=rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)

# Evaluasi Model Random Forest
print("=== Random Forest Classifier Report (Tuned) ===")
rf_best = rf_grid.best_estimator_
rf_preds = rf_best.predict(X_test)
print(classification_report(y_test, rf_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

# GridSearch untuk XGBoost
xgb_param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

xgb_clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_grid = GridSearchCV(estimator=xgb_clf, param_grid=xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
xgb_grid.fit(X_train, y_train)

# Evaluasi Model XGBoost
print("\n=== XGBoost Classifier Report (Tuned) ===")
xgb_best = xgb_grid.best_estimator_
xgb_preds = xgb_best.predict(X_test)
print(classification_report(y_test, xgb_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, xgb_preds))