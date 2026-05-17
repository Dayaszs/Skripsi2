import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Nilai Siswa Smt 2", page_icon="🎓", layout="wide")

# --- 2. LOAD MODEL DARI LOKAL (.pkl) ---
@st.cache_resource
def load_models():
    model_path = 'prediksi_nilai_siswa_models.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"File {model_path} tidak ditemukan! Pastikan file berada di folder yang sama dengan script Streamlit.")
        return None

models_data = load_models()

# --- 3. KAMUS MAPPING (Hanya yang digunakan oleh top features) ---
pendidikan_map = {
    'Tidak Sekolah': 0, 'SD/Sederajat': 1, 'SMP/Sederajat': 2,
    'SMA/SMK/Sederajat': 3, 'Diploma (D1/D2/D3)': 4,
    'Sarjana (S1)': 5, 'Pascasarjana (S2)': 6, 'Doktor (S3)': 7
}
penghasilan_map = {
    '< Rp 2.500.000': 0, 'Rp 2.500.000 - Rp 4.000.000': 1,
    'Rp 4.000.001 - Rp 8.000.000': 2, 'Rp 8.000.001 - Rp 10.000.000': 3,
    '> Rp 10.000.000': 4
}
ukuran_keluarga_map = {'1 Orang': 1, '2-3 Orang': 2, '4-5 Orang': 3, '6 Orang atau lebih': 4}
waktu_tempuh_map = {'Kurang dari 15 menit': 0, '15 - 30 menit': 1, '31 - 60 menit': 2, 'Lebih dari 60 menit': 3}

# --- KONFIGURASI MODEL TERBAIK ---
best_models_keys = {
    'Matematika': 'pipeline_RFR',       # R2: 0.7948
    'Bahasa Inggris': 'pipeline_SVR',   # R2: 0.8397
    'Bahasa Indonesia': 'pipeline_RFR'  # R2: 0.7075
}

# --- 4. TAMPILAN ANTARMUKA (UI) ---
st.title("🎓 Prediksi Nilai Semester 2 dengan AI")
st.markdown("Aplikasi ini secara otomatis menggunakan model **Linear Regression** untuk Matematika, **SVR** untuk B. Inggris, dan **Random Forest** untuk B. Indonesia berdasarkan performa terbaiknya.")
st.markdown("---")

if models_data:
    with st.form("prediction_form"):
        st.subheader("📝 Input Data Historis & Profil Siswa")
        
        col1, col2, col3 = st.columns(3)
        
        # --- BLOK 1: NILAI SEMESTER 1 ---
        with col1:
            st.markdown("**📊 Nilai Lintas Semester (Kelas X Smt 1)**")
            nilai_mtk = st.number_input("Nilai Matematika (Smt 1)", min_value=0, max_value=100, value=80)
            nilai_ing = st.number_input("Nilai Bahasa Inggris (Smt 1)", min_value=0, max_value=100, value=80)
            nilai_ind = st.number_input("Nilai Bahasa Indonesia (Smt 1)", min_value=0, max_value=100, value=80)
            
            st.markdown("**🧠 Motivasi & Sikap**")
            motivasi = st.slider("Motivasi Belajar Pribadi (Skala 1-5)", 1, 5, 3)
            dukungan = st.slider("Dukungan Orang Tua (Skala 1-5)", 1, 5, 3)

        # --- BLOK 2: KEBIASAAN BELAJAR ---
        with col2:
            st.markdown("**📚 Kebiasaan & Kedisiplinan**")
            jam_belajar = st.number_input("Jam Belajar di Luar Sekolah (Jam/Minggu)", min_value=0, max_value=100, value=10)
            absen = st.number_input("Jumlah Ketidakhadiran (Hari)", min_value=0, max_value=50, value=0)
            ekskul = st.number_input("Jumlah Ekstrakurikuler yang diikuti", min_value=0, max_value=10, value=1)
            waktu_tempuh = st.selectbox("Waktu Tempuh ke Sekolah", list(waktu_tempuh_map.keys()))

        # --- BLOK 3: LATAR BELAKANG KELUARGA ---
        with col3:
            st.markdown("**🏡 Latar Belakang Keluarga**")
            pend_ayah = st.selectbox("Pendidikan Terakhir Ayah", list(pendidikan_map.keys()))
            pend_ibu = st.selectbox("Pendidikan Terakhir Ibu", list(pendidikan_map.keys()))
            penghasilan = st.selectbox("Total Penghasilan Ortu/Bulan", list(penghasilan_map.keys()))
            keluarga = st.selectbox("Ukuran Keluarga Inti (Serumah)", list(ukuran_keluarga_map.keys()))

        st.markdown("---")
        submit_button = st.form_submit_button(label="✨ Hitung Prediksi Nilai Semester 2")

    # --- 5. LOGIKA PREDIKSI ---
    if submit_button:
        with st.spinner("Memproses dengan model AI terbaik..."):
            # 1. Tampung input mentah & mapping ke format model
            mapped_input = {
                'Nilai Matematika (Kelas X Semester 1)': nilai_mtk,
                'Nilai Bahasa Inggris (Kelas X Semester 1)': nilai_ing,
                'Nilai Bahasa Indonesia (Kelas X Semester 1)': nilai_ind,
                'Jam Belajar/Minggu': jam_belajar,
                'Jumlah ketidakhadiran pada saat kelas X Semester 2': absen,
                'Jumlah Ekstrakurikuler (Total berapa kegiatan ekstrakurikuler (Ekskul) yang Anda ikuti pada saat kelas X Semester 2?)': ekskul,
                'Pendidikan terakhir Ayah': pendidikan_map[pend_ayah],
                'Pendidikan terakhir Ibu': pendidikan_map[pend_ibu],
                'Total penghasilan (per bulan) kedua orang tua pada saat kelas X': penghasilan_map[penghasilan],
                'Ukuran keluarga inti pada saat kelas X (Total berapa anggota keluarga inti (termasuk Anda) yang tinggal serumah?)': ukuran_keluarga_map[keluarga],
                'Waktu tempuh dari rumah ke sekolah Pada Saat Kelas X (Berapa perkiraan durasi perjalanan dari rumah hingga sampai di sekolah?)': waktu_tempuh_map[waktu_tempuh],
                'Motivasi Belajar Pribadi pada saat kelas X': motivasi,
                'Tingkat dukungan orang tua terhadap pendidikan Anda pada saat kelas X': dukungan
            }
            
            hasil_prediksi = {}
            targets = ['Matematika', 'Bahasa Inggris', 'Bahasa Indonesia']
            
            # 2. Iterasi Prediksi per Mata Pelajaran
            for mapel in targets:
                target_data = models_data[mapel]
                fitur_terpilih = target_data['selected_features']
                
                # Dinamis mengambil model terbaik yang sudah ditentukan di atas
                selected_model_key = best_models_keys[mapel]
                model_prediksi = target_data[selected_model_key]
                
                # Buat DataFrame kosong khusus untuk fitur yang dibutuhkan (10, 8, atau 11 fitur)
                df_X = pd.DataFrame(columns=fitur_terpilih)
                df_X.loc[0] = 0 # Inisialisasi baris pertama
                
                # Isi nilai dari mapped_input ke kolom yang sesuai
                for col in df_X.columns:
                    if col in mapped_input:
                        df_X.at[0, col] = mapped_input[col]
                
                # Lakukan Prediksi
                pred_val = model_prediksi.predict(df_X)[0]
                hasil_prediksi[mapel] = np.clip(pred_val, 0, 100) # Pastikan rentang nilai masuk akal 0-100
            
            st.session_state['hasil_prediksi'] = hasil_prediksi
            st.session_state['prediksi_selesai'] = True

    # --- 6. TAMPILKAN HASIL ---
    if st.session_state.get('prediksi_selesai', False):
        st.success("✅ Analisis Berhasil! Berikut adalah proyeksi nilai siswa tersebut di Semester 2:")
        
        hasil = st.session_state['hasil_prediksi']
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric(label="📐 Prediksi Matematika (Model RFR)", value=f"{hasil['Matematika']:.2f}")
        col_m2.metric(label="🗣️ Prediksi B. Inggris (Model SVR)", value=f"{hasil['Bahasa Inggris']:.2f}")
        col_m3.metric(label="📖 Prediksi B. Indonesia (Model RFR)", value=f"{hasil['Bahasa Indonesia']:.2f}")
        
        st.markdown("---")
        st.caption("🔍 **Catatan**: Nilai historis Semester 1 menyumbang faktor prediksi terbesar (>70%), diikuti oleh kedisiplinan belajar (Jam belajar & Kehadiran).")