#!/usr/bin/env python
# coding: utf-8

# In[252]:


import pandas as pd


# In[253]:


# Baca data
df = pd.read_csv('loan_data_2007_2014.csv')
df.head()


# ### Data Understanding

# In[254]:


# Cek ada kolom apa saja
print(df.columns)

# Cek jumlah semua kolom yang ada
print('\nJumlah kolom pada dataset ada', len(df.columns), 'kolom')


# In[255]:


# Cek info semua kolom
df.info()


# In[256]:


# cek kolom yang semua barisnya null
null_cols = df.columns[df.isnull().all()]
print("Kolom yang semua barisnya null:", null_cols.tolist())
print("\nJumlah kolom yang barisnya null semua:", len(null_cols))


# #### Berdasarkan pengecekan, jumlah semua kolom ada 75 kolom, jumlah baris ada 466285 data, dan ada 17 kolom yang datanya kosong semua (null/NaN). Maka 17 kolom tersebut akan kita sisihkan.

# In[257]:


# bikin df baru tanpa kolom yang full null
df_new = df.dropna(axis=1, how='all')

print("Jumlah kolom awal:", df.shape[1])
print("Jumlah kolom setelah dibuang:", df_new.shape[1])


# In[258]:


df_new.head()


# In[259]:


# Cek lagi info kolom df baru
df_new.info()


# #### Berdasarkan kolom df baru, setelah melihat data dictionary, maka hanya dipakai beberapa kolom saja yang relevan dengan analisis risiko kredit. Sedangkan kolom yang kurang relevan tidak akan dipakai. Selain itu, kita dapat membaginya menjadi 3 kategori, yaitu: pinjaman, peminjam, dan kredit.

# Kategori Pinjaman

# In[260]:


# Kategori pinjaman
kategori_pinjaman = [
    "loan_amnt", "funded_amnt", "funded_amnt_inv",
    "term", "int_rate", "installment",
    "grade", "sub_grade", "purpose",
    "issue_d", "initial_list_status", "loan_status"
]
df_pinjaman = df_new[kategori_pinjaman]
print("Shape df_pinjaman:", df_pinjaman.shape)
df_pinjaman.head()


# In[261]:


df_pinjaman.info()


# Kategori Peminjam

# In[262]:


# Kategori peminjam
kategori_peminjam = [
    "emp_title", "emp_length",
    "home_ownership", "annual_inc",
    "application_type", "verification_status",
    "addr_state"
]
df_peminjam = df_new[kategori_peminjam]
print("Shape df_peminjam:", df_peminjam.shape)
df_peminjam.head()


# In[263]:


df_peminjam.info()


# Kategori Kredit

# In[264]:


# Kategori kredit
kategori_kredit = [
    "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "total_acc", "pub_rec",
    "revol_bal", "revol_util",
    "mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog",
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim"
]
df_kredit = df_new[kategori_kredit]
print("Shape df_kredit:", df_kredit.shape)
df_kredit.head()


# In[265]:


df_kredit.info()


# #### Setelah dibagi jadi 3 kategori, kita mendapatkan kolom-kolom yang relevan untuk analisis risiko kredit. Kemudian kita gabungkan kembali semua kolom pada 3 kategori itu untuk mendapat semua kolom yang relevan.

# In[266]:


# gabungkan ketiga dataframe jadi satu tabel
df_gabung = pd.concat([df_pinjaman, df_peminjam, df_kredit], axis=1)

# cek hasil
print(df_gabung.shape)
df_gabung.head()


# In[267]:


# Cek lagi info data yang sudah digabung
df_gabung.info()


# In[268]:


df_gabung.describe()


# Cek Missing Value

# In[269]:


# Cek missing value
print(df_gabung.isnull().sum())


# In[270]:


# Persentase missing value
(df_gabung.isnull().sum()/len(df_gabung)).to_frame('Persentase Missing')


# Cek Outlier

# In[271]:


# Cek dengan boxplot
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns

# --- Numerik ---
num_cols = df_gabung.select_dtypes(include=['int64', 'float64']).columns

# Tentukan ukuran figure biar rapi
plt.figure(figsize=(15, 8))

# Bikin boxplot untuk semua kolom numerik
df_gabung[num_cols].boxplot(rot=90)  # rot=90 biar nama kolom miring kebawah
plt.title("Boxplot Semua Kolom Numerik untuk Deteksi Outlier")
plt.show()


# In[272]:


# Ambil semua kolom numerik 
num_cols = df_gabung.select_dtypes(include=['int64', 'float64']).columns

# Dictionary buat nyimpen jumlah outlier tiap kolom
outlier_counts = {}

for col in num_cols:
    Q1 = df_gabung[col].quantile(0.25)
    Q3 = df_gabung[col].quantile(0.75)
    IQR = Q3 - Q1

    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR

    outliers = df_gabung[(df_gabung[col] < batas_bawah) | (df_gabung[col] > batas_atas)]
    outlier_counts[col] = outliers.shape[0]

# Tampilkan hasil
for col, count in outlier_counts.items():
    print(f"Jumlah outlier di {col}: {count}")


# ### Exploratory Data Analysis

# In[273]:


# Cek lagi info data yang sudah digabung
df_gabung.info()


# In[176]:


import matplotlib.pyplot as plt
import seaborn as sns

# --- Numerik ---
num_cols = df_gabung.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    plt.figure(figsize=(12,5))

    # Histogram
    plt.subplot(1,2,1)
    sns.histplot(df_gabung[col], bins=30, kde=True, color='blue')
    plt.title(f'Distribusi {col}')

    # Boxplot
    plt.subplot(1,2,2)
    sns.boxplot(x=df_gabung[col], color='orange')
    plt.title(f'Boxplot {col}')

    plt.tight_layout()
    plt.show()

# --- Kategorikal ---
cat_cols = df_gabung.select_dtypes(include=['object']).columns

for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(y=col, data=df_gabung, order=df[col].value_counts().index, palette="viridis")
    plt.title(f'Frekuensi {col}')
    plt.show()


# In[178]:


# Hitung korelasi numerik
corr = df_gabung[num_cols].corr()

# Visualisasi heatmap
plt.figure(figsize=(14,10))
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
plt.title('Heatmap Korelasi Antar Variabel Numerik')
plt.show()


# Berdasarkan heatmap di atas, dapat disimpulkan korelasi antar variabel berikut.
# 
# ##### 1. Hubungan sangat kuat (positif mendekati 1)
# 
# Kolom *loan_amnt*, *funded_amnt*, *funded_amnt_inv*, dan *installment*, keempatnya saling berkorelasi sangat tinggi. Logis, karena jumlah pinjaman, pendanaan, dan cicilan memang saling terkait. Selain itu, kolom *total_rev_hi_lim* dengan *revol_bal* juga ada korelasi cukup kuat, karena limit kartu kredit tinggi biasanya diikuti saldo pemakaian besar.
# 
# ##### 2. Hubungan sedang (positif 0.4 – 0.6)
# 
# Kolom *open_acc* dengan *total_acc* memiliki hubungan positif sedang. Wajar, karena makin banyak akun aktif makin besar total akun. Lalu, kolom *tot_cur_bal* dengan *total_rev_hi_lim* juga memiliki kaitan, saldo total cenderung lebih besar kalau limit kredit juga besar.
# 
# ##### 3. Hubungan lemah sampai hampir tidak ada (sekitar -0.2 sampai 0.2)
# 
# Kolom *annual_inc* dengan mayoritas variabel lain memiliki hubungan yang lemah, artinya penghasilan tahunan tidak terlalu linear dengan besaran pinjaman atau cicilan. Kemudian, kolom *dti*, *inq_last_6mths*, *mths_since_last_delinq*, dan beberapa variabel riwayat kredit cenderung berdiri sendiri, korelasinya rendah dengan variabel lain.
# 
# ##### 4. Hubungan negatif (meski relatif lemah)
# 
# Ada beberapa kombinasi yang agak kebiruan (misalnya antara variabel *inquiry kredit* dan *waktu sejak keterlambatan*), tapi kekuatannya kecil, artinya bukan hubungan yang dominan.

# ### Data Preparation

# Menangani missing value

# Karena banyaknya missing value di beberapa kolom yang bahkan persentasenya melebihi 50% data, maka missing value ditangani dengan imputasi statistik, seperti mean, modus, atau median.

# In[274]:


# Tangani missing value untuk kolom numerik (pakai mean/rata-rata)
for col in num_cols:
    df_gabung[col] = df_gabung[col].fillna(df_gabung[col].mean())


# In[275]:


# Tangani missing value untuk kolom kategorikal (pakai modus)
for col in cat_cols:
    df_gabung[col] = df_gabung[col].fillna(df_gabung[col].mode()[0])


# In[276]:


# Cek missing value lagi
print(df_gabung.isnull().sum())


# Menangani Outlier

# In[277]:


# Beri batas nilai
import numpy as np

def winsorize_series(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR
    return np.where(series < batas_bawah, batas_bawah,
           np.where(series > batas_atas, batas_atas, series))

# Terapkan ke kolom numerik
for col in num_cols:
    df_gabung[col] = winsorize_series(df_gabung[col])


# In[278]:


# Cek lagi outlier
num_cols = df_gabung.select_dtypes(include=['int64', 'float64']).columns

# Dictionary buat nyimpen jumlah outlier tiap kolom
outlier_counts = {}

for col in num_cols:
    Q1 = df_gabung[col].quantile(0.25)
    Q3 = df_gabung[col].quantile(0.75)
    IQR = Q3 - Q1

    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR

    outliers = df_gabung[(df_gabung[col] < batas_bawah) | (df_gabung[col] > batas_atas)]
    outlier_counts[col] = outliers.shape[0]

# Tampilkan hasil
for col, count in outlier_counts.items():
    print(f"Jumlah outlier di {col}: {count}")


# Apakah Perlu Encoding?

# Cek berapa kategori untuk masing-masing kolom kategori

# In[279]:


import pandas as pd

# cek jumlah kategori unik tiap kolom object
cat_cols = df_gabung.select_dtypes(include='object').columns

for col in cat_cols:
    n_unique = df_gabung[col].nunique()
    print(f"{col}: {n_unique} kategori")


# In[280]:


# Lihat data dulu
import pandas as pd

# Tampilkan semua kolom tanpa batas
pd.set_option('display.max_columns', None)

# Kalau mau semua baris juga kelihatan:
pd.set_option('display.max_rows', None)

# Contoh tampilkan data
print(df_gabung.head())  # atau df.tail(), df.sample(5), dll.


# In[281]:


for col in df_gabung.columns:
    if df_gabung[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_gabung[col]):
        unique_vals = df_gabung[col].dropna().unique()
        if len(unique_vals) < 10:
            print(f"Kolom: {col} ({len(unique_vals)} kategori)")
            print("→", unique_vals)
            print("-" * 40)


# Encoding kolom kategorikal diperlukan agar model bisa mengenali data dengan baik

# In[283]:


import pandas as pd

# Ubah issue_d ke fitur waktu
df_gabung['issue_d'] = pd.to_datetime(df_gabung['issue_d'], format='%b-%y', errors='coerce')
df_gabung['issue_month'] = df_gabung['issue_d'].dt.month
df_gabung['issue_year'] = df_gabung['issue_d'].dt.year


# In[284]:


# Encode emp_title → top 20 + Other
top_titles = df_gabung['emp_title'].value_counts().nlargest(20).index
df_gabung['emp_title_clean'] = df_gabung['emp_title'].apply(lambda x: x if x in top_titles else 'Other')


# In[285]:


# Encode addr_state → frequency encoding
state_freq = df_gabung['addr_state'].value_counts(normalize=True)
df_gabung['addr_state_freq'] = df['addr_state'].map(state_freq)


# In[286]:


# Encode emp_length → ordinal
emp_length_map = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10
}
df_gabung['emp_length_encoded'] = df_gabung['emp_length'].map(emp_length_map)


# In[288]:


# Encode grade → ordinal (A–G → 0–6)
grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
df_gabung['grade_encoded'] = df_gabung['grade'].map({g: i for i, g in enumerate(grade_order)})


# In[289]:


# Encode sub_grade → ordinal (A1–G5 → 0–34)
subgrade_order = [f"{g}{n}" for g in grade_order for n in range(1, 6)]
df_gabung['sub_grade_encoded'] = df_gabung['sub_grade'].map({sg: i for i, sg in enumerate(subgrade_order)})


# In[293]:


# One-Hot Encoding kolom dengan kategori kecil
onehot_cols = ['term', 'purpose', 'initial_list_status', 'home_ownership', 'verification_status', 'emp_title_clean']
df_gabung = pd.get_dummies(df_gabung, columns=onehot_cols, drop_first=True)


# In[294]:


# Drop kolom yang sudah di-encode
df_gabung = df_gabung.drop(columns=[
    'grade', 'sub_grade', 'emp_length', 'emp_title', 'addr_state', 'issue_d',
    'application_type'  # karena tidak ada variasi
])


# In[295]:


# Cek dahulu
print(df_gabung.head())


# Encoding kolom loan_status sebagai label data

# In[296]:


df_gabung['loan_status_binary'] = df_gabung['loan_status'].apply(
    lambda x: 1 if x == 'Fully Paid' else 0
)


# In[297]:


print(df_gabung.head())


# Scalling data numerik

# In[298]:


from sklearn.preprocessing import StandardScaler

# Daftar kolom numerik
numerik_cols = [
    'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment',
    'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
    'total_acc', 'pub_rec', 'revol_bal', 'revol_util',
    'mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog',
    'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim'
]

# Scaling
scaler = StandardScaler()
df_gabung[numerik_cols] = scaler.fit_transform(df_gabung[numerik_cols])


# In[299]:


print(df_gabung.head())


# Pisahkan fitur dan label

# In[300]:


X = df_gabung.drop(columns=['loan_status', 'loan_status_binary'])  # fitur
y = df_gabung['loan_status_binary']  # label


# Split data untuk training dan testing

# In[301]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Data Modelling dan Evaluasi Model

# #### Logistic Regression

# In[305]:


print(X_train.shape)  # Berapa banyak fitur?


# In[306]:


# Latih model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='saga', max_iter=10000)
model.fit(X_train, y_train)


# Evaluasi Model Logistic Regression

# In[307]:


from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Berdasarkan hasil akurasi, didapat bahwa:
# - Precision: Dari semua prediksi class 1 (Charged Off), hanya 64% yang benar. Artinya model kadang salah nuduh pinjaman gagal padahal lunas.
# - Recall: Dari semua pinjaman yang benar-benar gagal, hanya 43% yang berhasil ditangkap model. Ini agak rendah → model sering miss kasus gagal bayar.
# - F1-score: Kombinasi precision dan recall. Nilai 0.51 untuk class 1 menunjukkan performa sedang.
# - Support: Jumlah data aktual di masing-masing class.
# 
# Sehingga:
# - Model lebih jago mengenali pinjaman lunas (class 0) daripada yang gagal (class 1).
# - Cocok untuk baseline, tapi perlu ditingkatkan kalau tujuan kamu adalah mendeteksi risiko gagal bayar.
# 

# In[308]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Berdasarkan hasil confussion matriks, didapat:
# - True Positives (TP): 15,899 → model berhasil prediksi pinjaman gagal
# - False Negatives (FN): 21,007 → model salah prediksi gagal jadi lunas
# - True Negatives (TN): 47,238 → model benar prediksi lunas
# - False Positives (FP): 9,113 → model salah prediksi lunas jadi gagal
# 
# Sehingga:
# - Model cenderung “main aman” → lebih sering bilang pinjaman itu lunas
# - Banyak kasus gagal yang lolos → ini bisa berisiko kalau dipakai buat screening kredit

# In[309]:


import pandas as pd

coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("Fitur yang mendorong ke Fully Paid:")
print(coef_df.head(10))

print("\nFitur yang mendorong ke Charged Off:")
print(coef_df.tail(10))


# Berdasarkan hasil di atas, didapat:
# Fitur yang Mendorong ke Fully Paid
# - int_rate: bunga rendah → lebih mungkin lunas
# - tot_cur_bal, total_rev_hi_lim: saldo dan limit tinggi → kemampuan bayar lebih besar
# - emp_title_clean_Other: bisa jadi noise atau efek generalisasi
# - purpose_wedding: mungkin peminjam dengan tujuan ini punya profil keuangan lebih stabil
# 
# Fitur yang Mendorong ke Charged Off
# - term_60 months: tenor panjang → risiko gagal lebih tinggi
# - annual_inc: pendapatan rendah → kemampuan bayar terbatas
# - sub_grade_encoded, initial_list_status_w: kualitas kredit rendah
# - verification_status_Source Verified: bisa jadi kurang validasi internal
# 
# Sehingga:
# - Fitur-fitur keuangan seperti bunga, saldo, dan limit punya pengaruh besar
# - Tujuan pinjaman dan status verifikasi juga berkontribusi
# - Model bisa dipakai untuk analisis risiko, tapi perlu ditingkatkan untuk deteksi gagal bayar
# 

# Kesimpulan dari model Logistic Regression adalah sebagai berikut.
# - Model kamu punya akurasi 67.7%, cukup baik untuk baseline
# - Tapi recall untuk pinjaman gagal masih rendah (43%)
# - Cocok untuk analisis awal dan interpretasi, tapi belum ideal untuk keputusan kredit otomatis

# #### Random Forest

# In[310]:


# Latih model
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # penting untuk menangani class imbalance
    random_state=42,
    n_jobs=-1  # biar training lebih cepat
)

model_rf.fit(X_train, y_train)


# Evaluasi Model Random Forest

# In[311]:


# Evaluasi performa
from sklearn.metrics import accuracy_score, classification_report

y_pred_rf = model_rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# Berdasarkan hasil di atas, didapat:
# - Precision class 1 (Charged Off) = 0.77 → dari semua prediksi gagal bayar, 77% benar
# - Recall class 1 = 0.46 → dari semua pinjaman yang benar-benar gagal, hanya 46% yang berhasil ditangkap model
# - F1-score class 1 = 0.57 → keseimbangan antara precision dan recall, lebih baik dari Logistic Regression sebelumnya (0.51)
# - Accuracy total = 73.1% → naik dari 67.7% sebelumnya
# 
# Sehingga:
# Model Random Forest lebih akurat dan lebih seimbang dibanding Logistic Regression. Recall untuk class 1 meningkat dari 43% → 46%, dan precision-nya jauh lebih tinggi.

# In[313]:


# Confussion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()


# Berdasarkan hasil confussion matrix di atas, didapat:
# - True Positives (TP): 16,827 → model berhasil prediksi gagal bayar
# - False Negatives (FN): 20,079 → model masih miss cukup banyak pinjaman gagal
# - False Positives (FP): 4,982 → model kadang nuduh gagal padahal lunas, tapi lebih sedikit dari Logistic Regression
# 
# Sehingga:
# Random Forest lebih konservatif dan lebih akurat dalam mengenali pinjaman lunas, tapi tetap ada ruang untuk meningkatkan recall class 1.

# In[314]:


# Feature Importance
import pandas as pd

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importance.head(10))  # fitur paling berpengaruh


# Berdasarkan hasil di atas, didapat:
# - Fitur paling berpengaruh adalah issue_year → tahun pinjaman dikeluarkan punya dampak besar (mungkin karena tren ekonomi)
# - Fitur keuangan seperti saldo, limit, pendapatan, bunga, dan cicilan sangat menentukan outcome
# - Ini menunjukkan model menangkap pola dari kondisi finansial peminjam
# 
# Sehingga:
# Model Random Forest:
# - Akurasi lebih tinggi (73% vs 67%)
# - Precision class 1 sangat baik (77%)
# - Bisa menangani data besar dan fitur banyak tanpa scaling
# - Memberikan insight lewat feature importance
# 
# Tetapi :
# - Recall class 1 masih moderat (46%) → masih banyak pinjaman gagal yang lolos
# - Interpretasi tidak sejelas Logistic Regression (karena model non-linear)
# 
# 
