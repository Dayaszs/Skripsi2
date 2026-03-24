import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Nilai 5 Mata Pelajaran", page_icon="🎓", layout="centered")

# --- 2. LOAD ARTEFAK DARI GOOGLE DRIVE ---
@st.cache_resource
def load_all_artifacts():
    model_dir = "downloaded_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Masukkan Google Drive File ID untuk 5 Model dan 5 Preprocessor
    # Ganti 'TULIS_ID_DISINI' dengan ID asli
    drive_files = {
        # Model
        'RF_Math.pkl': '188wiqIRtTf-Emgm6IY5vim3bZI7yO9TA',
        'RF_Science.pkl': '1IZXr14--RQTCGIi7E-bnhqqgHvqy0-Rc',
        'RF_Social.pkl': '1JggbONCcrDt106F7y4TxLj-TpSkISs2E',
        'RF_English.pkl': '1URxlRgfADjdFiGGozS5gIB-6qE6wDIO8',
        'RF_Art.pkl': '1Bct3rQsdnG_0ixa7Weg11qJw1gEYAhL6',
        # Preprocessor masing-masing pelajaran
        'prep_Math.pkl': '1dKXF9sb_yXMyafYgY7_PO2WwT6YD8T0n',
        'prep_Science.pkl': '1vrEfs6ljY3wyGB_D3Bi86YP1TvmvA0D_',
        'prep_Social.pkl': '1XlqeMlxJmA01Rj6AGYS4nh2TWfLSH1nj',
        'prep_English.pkl': '1-5RQAw3CATJyxzjvVneXxk2htWFVbyhU',
        'prep_Art.pkl': '1pmB_M1cmalUGYTUd31HYspbL0jReqvmU'
    }

    # Download setiap file jika belum ada
    for filename, file_id in drive_files.items():
        filepath = os.path.join(model_dir, filename)
        if not os.path.exists(filepath) and file_id != 'TULIS_ID_DISINI':
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filepath, quiet=False)

    # Load Model dan Preprocessor ke dalam dictionary terpisah
    subjects = ['Math', 'Science', 'Social', 'English', 'Art']
    models = {}
    preprocessors = {}
    
    for subj in subjects:
        try:
            models[subj] = joblib.load(os.path.join(model_dir, f'RF_{subj}.pkl'))
            preprocessors[subj] = joblib.load(os.path.join(model_dir, f'prep_{subj}.pkl'))
        except FileNotFoundError:
            # Lewati jika file belum didownload (misal ID belum diisi)
            pass
            
    return models, preprocessors

with st.spinner("Mengunduh dan memuat model beserta preprocessor..."):
    models, preprocessors = load_all_artifacts()

# --- 3. DAFTAR KATEGORI & FITUR LENGKAP ---
categorical_options = {
    'mother_education': ['Non_Educated', 'Under_SSC', 'SSC', 'HSC', 'Diploma', 'Honors', 'Masters'],
    'father_education': ['Non_Educated', 'Under_SSC', 'SSC', 'HSC', 'Diploma', 'Honors', 'Masters'],
    'gender': ['Male', 'Female'],
    'location': ['urban', 'rural', 'city'],
    'mother_job': ['No', 'Yes'],
    'father_job': ['No', 'Yes'],
    'guardian': ['Father', 'Mother', 'Other'],
    'parental_involvement': ['Yes', 'No'],
    'internet_access': ['Yes', 'No'],
    'tutoring': ['Yes', 'No'],
    'school_type': ['Private', 'Semi_Govt', 'Govt'],
    'extra_curricular_activities': ['Yes', 'No'],
    'stu_group': ['Science', 'Commerce', 'Arts']
}

numerical_features_list = ['age', 'family_size', 'studytime', 'attendance']

# Gabungkan semua fitur karena tidak pakai selected_features lagi
all_features = numerical_features_list + list(categorical_options.keys())

# --- 4. TAMPILAN ANTARMUKA (UI) ---
st.title("🎓 Sistem Prediksi Nilai Akademik Siswa")
st.markdown("Masukkan data siswa di bawah ini untuk memprediksi nilai pada 5 mata pelajaran (Math, Science, Social Science, English, Art & Culture).")
st.markdown("---")

with st.form("prediction_form"):
    st.subheader("Input Data Siswa")
    user_input = {}
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(all_features):
        current_col = col1 if i % 2 == 0 else col2
        
        with current_col:
            if feature in numerical_features_list:
                if feature == 'age':
                    user_input[feature] = st.number_input("Umur (age)", min_value=10, max_value=30, value=15)
                elif feature == 'attendance':
                    user_input[feature] = st.number_input("Kehadiran % (attendance)", min_value=0.0, max_value=100.0, value=80.0)
                elif feature == 'studytime':
                    user_input[feature] = st.number_input("Waktu Belajar/Minggu (studytime)", min_value=0.0, max_value=50.0, value=5.0)
                elif feature == 'family_size':
                    user_input[feature] = st.number_input("Jumlah Anggota Keluarga (family_size)", min_value=1, max_value=20, value=4)
                else:
                    user_input[feature] = st.number_input(f"{feature}", value=0.0)
            else:
                options = categorical_options.get(feature, ['Yes', 'No'])
                user_input[feature] = st.selectbox(f"Pilih {feature.replace('_', ' ').title()}", options)
                
    st.markdown("---")
    submit_button = st.form_submit_button(label="🔍 Mulai Prediksi")

# --- 5. LOGIKA PREDIKSI & HASIL ---
if 'prediksi_selesai' not in st.session_state:
    st.session_state['prediksi_selesai'] = False
    st.session_state['hasil_prediksi'] = {}

if submit_button:
    # Cek apakah model dan preprocessor sudah berhasil diload (ID Drive valid)
    if len(models) < 5 or len(preprocessors) < 5:
         st.error("⚠️ Model atau Preprocessor belum lengkap! Pastikan Anda sudah memasukkan semua ID Google Drive di dalam kode.")
    else:
        with st.spinner("Menghitung prediksi 5 mata pelajaran..."):
            try:
                # 1. Format input ke DataFrame
                df_input = pd.DataFrame([user_input])
                hasil = {}
                
                # Nama display untuk UI
                display_names = {
                    'Math': 'Math',
                    'Science': 'Science',
                    'Social': 'Social Science',
                    'English': 'English',
                    'Art': 'Art & Culture'
                }
                
                # 2. Looping per mata pelajaran: Transform -> Predict
                for subj in models.keys():
                    prep = preprocessors[subj]
                    model = models[subj]
                    
                    # Transform input dengan preprocessor spesifik pelajaran ini
                    X_proc = prep.transform(df_input)
                    
                    # Prediksi
                    pred_raw = model.predict(X_proc)[0]
                    pred_clipped = np.clip(pred_raw, 0, 100)
                    
                    hasil[display_names[subj]] = round(pred_clipped, 2)
                
                st.session_state['hasil_prediksi'] = hasil
                st.session_state['prediksi_selesai'] = True
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat memproses data: {e}")

# --- 6. TAMPILKAN HASIL JIKA PREDIKSI SUDAH DILAKUKAN ---
if st.session_state['prediksi_selesai']:
    st.success("Prediksi berhasil dilakukan!")
    
    st.markdown("### 🏆 Estimasi Nilai Siswa")
    hasil = st.session_state['hasil_prediksi']
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric(label="📘 Math", value=f"{hasil['Math']} / 100")
    col_b.metric(label="🧪 Science", value=f"{hasil['Science']} / 100")
    col_c.metric(label="🌍 Social Science", value=f"{hasil['Social Science']} / 100")
    
    col_d, col_e = st.columns(2)
    col_d.metric(label="🗣️ English", value=f"{hasil['English']} / 100")
    col_e.metric(label="🎨 Art & Culture", value=f"{hasil['Art & Culture']} / 100")
    
    # --- INTERPRETASI MODEL (FEATURE IMPORTANCE) ---
    st.markdown("---")
    st.markdown("### 🔍 Interpretasi Model (Feature Importance)")
    
    pilihan_pelajaran = {
        'Math': 'Math',
        'Science': 'Science',
        'Social Science': 'Social',
        'English': 'English',
        'Art & Culture': 'Art'
    }
    
    selected_subject_display = st.selectbox("Pilih Mata Pelajaran untuk melihat faktor terpenting:", list(pilihan_pelajaran.keys()))
    subject_key = pilihan_pelajaran[selected_subject_display]
    
    if subject_key in models and subject_key in preprocessors:
        model_to_explain = models[subject_key]
        prep_to_explain = preprocessors[subject_key]
        
        # Ambil nama fitur yang sudah ditransformasi dari preprocessor spesifik ini
        encoded_feature_names = prep_to_explain.get_feature_names_out()
        weights = model_to_explain.feature_importances_ * 100
        
        df_weights = pd.DataFrame({
            "Fitur (Transformed)": encoded_feature_names,
            "Importance (%)": weights
        })
        
        df_weights = df_weights.sort_values(by='Importance (%)', ascending=False).reset_index(drop=True)
        df_weights.index = np.arange(1, len(df_weights) + 1)
        
        st.write(f"**10 Fitur Paling Berpengaruh untuk {selected_subject_display}:**")
        st.dataframe(df_weights.head(10).style.format({'Importance (%)': '{:.2f}%'}), width='stretch')