import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Nilai Ujian Siswa", page_icon="🎓", layout="centered")

# --- 2. LOAD ARTEFAK DARI GOOGLE DRIVE ---
@st.cache_resource
def load_all_artifacts():
    # Buat folder sementara untuk menyimpan model yang didownload
    model_dir = "downloaded_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Masukkan Google Drive File ID untuk masing-masing file
    # Ganti 'TULIS_ID_DISINI' dengan ID asli dari link Google Drive Anda
    drive_files = {
        'preprocessor_final.pkl': '16iN_If9u75u13U3BTsFlFWgCU9YPsXgH',
        'selected_features.pkl': '1vZbGmxtBh9H1xxoWjdzuVuKIgju84vmW',
        'Linear_Original.pkl': '1Nv9IkuzmKx3CB3Lw--QfQyS7n_0Jj4Gj',
        'RF_Original.pkl': '1H6SvdFR9T3VJQF9NhT8x86anu-DLwalQ',
        'SVR_Original.pkl': '1289_G6t9IejCbkGoaW1tvhVmO56AA-N7',
        'Linear_SMOGN.pkl': '120ntWVg-SWuGnmk2es9oh6JT1TbCQmZP',
        'RF_SMOGN.pkl': '1hRFfPfVS71OaY4hpHZ1-UpFCgUp5AOUq',
        'SVR_SMOGN.pkl': '1IHpzBRrCCiwBIzRcVTncsNz5Gq8kVko7',
        'Linear_SMOTER.pkl': '1PHybBfo2JtaBMeYlZQrGdWT6go2LAIsO',
        'RF_SMOTER.pkl': '1qK2pKaRgK9S71bU8sMKWHSEJbSeYdYXL',
        'SVR_SMOTER.pkl': '1y6kQZLdsc90wCX_Tu2sRJr1zUH2Ak1Cp'
    }

    # Download setiap file jika belum ada di folder lokal
    for filename, file_id in drive_files.items():
        filepath = os.path.join(model_dir, filename)
        if not os.path.exists(filepath):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filepath, quiet=False)

    # Load Preprocessor dan Features
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor_final.pkl'))
    selected_features = joblib.load(os.path.join(model_dir, 'selected_features.pkl'))
    
    # Daftar 9 kombinasi model
    model_names = [
        'Linear_Original', 'RF_Original', 'SVR_Original',
        'Linear_SMOGN', 'RF_SMOGN', 'SVR_SMOGN',
        'Linear_SMOTER', 'RF_SMOTER', 'SVR_SMOTER'
    ]
    
    # Load semua model ke dalam dictionary
    models = {}
    for name in model_names:
        models[name] = joblib.load(os.path.join(model_dir, f'{name}.pkl'))
        
    return models, preprocessor, selected_features

with st.spinner("Mengunduh dan memuat model... (Ini mungkin memakan waktu sebentar pada percobaan pertama)"):
    models, preprocessor, selected_features = load_all_artifacts()

# --- 3. DAFTAR KATEGORI SESUAI DATASET ---
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

# --- 4. TAMPILAN ANTARMUKA (UI) ---
st.title("🎓 Sistem Prediksi Nilai Ujian Siswa")
st.markdown("Masukkan data siswa di bawah ini untuk melihat perbandingan prediksi dari 9 model Machine Learning.")
st.markdown("---")

with st.form("prediction_form"):
    st.subheader("Input Data Siswa")
    user_input = {}
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(selected_features):
        current_col = col1 if i % 2 == 0 else col2
        
        with current_col:
            # Handle Numerical Features
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
            
            # Handle Categorical Features
            else:
                options = categorical_options.get(feature, ['Yes', 'No']) # Fallback default
                user_input[feature] = st.selectbox(f"Pilih {feature.replace('_', ' ').title()}", options)
                
    st.markdown("---")
    submit_button = st.form_submit_button(label="🔍 Mulai Prediksi")

# --- 5. LOGIKA PREDIKSI & HASIL ---

# Inisialisasi "memori" (session_state) agar hasil tidak hilang saat dropdown diganti
if 'prediksi_selesai' not in st.session_state:
    st.session_state['prediksi_selesai'] = False
    st.session_state['tabel_prediksi'] = None

if submit_button:
    with st.spinner("Menghitung prediksi dari ke-9 model..."):
        try:
            # 1. Format input ke DataFrame dan Transform
            df_input = pd.DataFrame([user_input])
            X_input_proc = pd.DataFrame(
                preprocessor.transform(df_input), 
                columns=preprocessor.get_feature_names_out()
            )
            
            # 2. Lakukan prediksi untuk semua model
            predictions = []
            for name, model in models.items():
                algo, data_type = name.split('_', 1)
                
                pred_raw = model.predict(X_input_proc)[0]
                pred_clipped = np.clip(pred_raw, 0, 100)
                
                predictions.append({
                    "Algoritma": algo,
                    "Dataset": data_type,
                    "Prediksi Nilai": round(pred_clipped, 2)
                })
            
            # 3. Buat DataFrame Hasil Prediksi dan Pivot agar rapi
            df_preds = pd.DataFrame(predictions)
            pivot_preds = df_preds.pivot(index="Algoritma", columns="Dataset", values="Prediksi Nilai")
            
            # SIMPAN HASIL KE MEMORI (SESSION STATE)
            st.session_state['tabel_prediksi'] = pivot_preds
            st.session_state['prediksi_selesai'] = True
            
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses data: {e}")

# --- TAMPILKAN HASIL JIKA PREDIKSI SUDAH DILAKUKAN ---
if st.session_state['prediksi_selesai']:
    st.success("Prediksi berhasil dilakukan!")
    st.markdown("### 📈 Perbandingan Hasil Prediksi (Skor 0-100)")
    
    # Ambil tabel dari memori
    st.dataframe(
        st.session_state['tabel_prediksi'].style.highlight_max(axis=None, color='lightgreen').highlight_min(axis=None, color='lightcoral'),
        width='stretch'
    )
    
    # --- TAMPILKAN FEATURE IMPORTANCE & KOEFISIEN ---
    st.markdown("---")
    st.markdown("### 🔍 Interpretasi Model (Feature Importance & Koefisien)")
    st.write("Pilih model di bawah ini untuk melihat fitur spesifik apa yang paling memengaruhi prediksinya.")
    
    # Dropdown di luar blok submit_button agar tidak hilang saat di-klik
    interpretable_models = [
        'RF_Original', 'RF_SMOGN', 'RF_SMOTER', 
        'Linear_Original', 'Linear_SMOGN', 'Linear_SMOTER'
    ]
    
    selected_interp_model = st.selectbox("Pilih Model:", interpretable_models)
    
    # Ambil model yang dipilih dari dictionary models
    model_to_explain = models.get(selected_interp_model)
    encoded_feature_names = preprocessor.get_feature_names_out()
    
    # Cek apakah ini model RF atau Linear
    if "RF" in selected_interp_model:
        weights = model_to_explain.feature_importances_
        col_name = "Importance (%)"
        weights = weights * 100  # Jadikan persentase
    elif "Linear" in selected_interp_model:
        weights = model_to_explain.coef_
        col_name = "Koefisien"
    
    # Buat DataFrame untuk interpretasi
    df_weights = pd.DataFrame({
        "Fitur (Transformed)": encoded_feature_names,
        col_name: weights
    })
    
    # Urutkan berdasarkan nilai absolut terbesar
    df_weights['Nilai Absolut'] = df_weights[col_name].abs()
    df_weights = df_weights.sort_values(by='Nilai Absolut', ascending=False).drop(columns=['Nilai Absolut']).reset_index(drop=True)
    df_weights.index = np.arange(1, len(df_weights) + 1)
    
    # Tampilkan 15 fitur teratas
    st.write(f"**Fitur Paling Berpengaruh pada {selected_interp_model}:**")
    
    if "RF" in selected_interp_model:
        st.dataframe(df_weights.head(15).style.format({col_name: '{:.2f}%'}), width='stretch')
        
    else:
        st.dataframe(df_weights.head(15).style.format({col_name: '{:.4f}'}), width='stretch')
        