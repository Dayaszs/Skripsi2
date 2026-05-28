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

# --- 3. KAMUS MAPPING ---
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

# --- 4. TAMPILAN ANTARMUKA (UI) ---
st.title("🎓 Prediksi Nilai Semester 2 dengan AI")
st.markdown("Berdasarkan evaluasi statistik, ketiga algoritma prediksi (MLR, SVR, RFR) memiliki performa yang setara dan sangat akurat (Tingkat kesalahan/MAPE < 2.5%). Aplikasi ini akan menampilkan proyeksi nilai dari ketiga algoritma tersebut sebagai bahan pertimbangan komprehensif.")
st.markdown("---")

if models_data:
    with st.form("prediction_form"):
        st.subheader("📝 Input Data Historis & Profil Siswa")
        
        col1, col2, col3 = st.columns(3)
        
        # --- BLOK 1: NILAI SEMESTER 1 ---
        with col1:
            st.markdown("**📊 Nilai Historis (Prediktor Absolut)**")
            nilai_mtk = st.number_input("Nilai Matematika (Kelas X Smt 1)", min_value=0, max_value=100, value=80)
            nilai_ing = st.number_input("Nilai Bahasa Inggris (Kelas X Smt 1)", min_value=0, max_value=100, value=80)
            nilai_ind = st.number_input("Nilai Bahasa Indonesia (Kelas X Smt 1)", min_value=0, max_value=100, value=80)

        # --- BLOK 2: KEBIASAAN BELAJAR ---
        with col2:
            st.markdown("**📚 Perilaku Belajar & Kedisiplinan**")
            jam_belajar = st.number_input("Jam Belajar di Luar Sekolah (Jam/Minggu)", min_value=0, max_value=100, value=10)
            absen = st.number_input("Jumlah Ketidakhadiran (Smt 2)", min_value=0, max_value=50, value=0)
            ekskul = st.number_input("Jumlah Ekstrakurikuler yang diikuti", min_value=0, max_value=10, value=1)
            waktu_tempuh = st.selectbox("Waktu Tempuh ke Sekolah", list(waktu_tempuh_map.keys()))

        # --- BLOK 3: LATAR BELAKANG KELUARGA ---
        with col3:
            st.markdown("**🏡 Latar Belakang Sosial-Ekonomi**")
            pend_ayah = st.selectbox("Pendidikan Terakhir Ayah", list(pendidikan_map.keys()))
            pend_ibu = st.selectbox("Pendidikan Terakhir Ibu", list(pendidikan_map.keys()))
            penghasilan = st.selectbox("Total Penghasilan Ortu/Bulan", list(penghasilan_map.keys()))
            keluarga = st.selectbox("Ukuran Keluarga Inti (Serumah)", list(ukuran_keluarga_map.keys()))

        st.markdown("---")
        submit_button = st.form_submit_button(label="✨ Hitung Prediksi Nilai Semester 2")

    # --- 5. LOGIKA PREDIKSI ---
    if submit_button:
        with st.spinner("Memproses dengan model komputasi MLR, SVR, dan RFR..."):
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
                'Waktu tempuh dari rumah ke sekolah Pada Saat Kelas X (Berapa perkiraan durasi perjalanan dari rumah hingga sampai di sekolah?)': waktu_tempuh_map[waktu_tempuh]
            }
            
            # Key aktual yang ada di dalam model .pkl Anda
            targets = ['Matematika_top_7', 'Bahasa Inggris_top_7', 'Bahasa Indonesia_top_7']
            
            # Dictionary untuk menerjemahkan nama file ke nama UI yang rapi
            display_names = {
                'Matematika_top_7': 'Matematika',
                'Bahasa Inggris_top_7': 'Bahasa Inggris',
                'Bahasa Indonesia_top_7': 'Bahasa Indonesia'
            }
            
            # Dictionary untuk menyimpan hasil semua model
            hasil_prediksi = {'Matematika': {}, 'Bahasa Inggris': {}, 'Bahasa Indonesia': {}}
            models_to_run = {'MLR': 'pipeline_MLR', 'SVR': 'pipeline_SVR', 'RFR': 'pipeline_RFR'}
            
            # 2. Iterasi Prediksi per Mata Pelajaran dan per Model
            for mapel_key in targets:
                target_data = models_data[mapel_key]
                fitur_terpilih = target_data['selected_features']
                nama_display = display_names[mapel_key] # Ambil nama bersih untuk UI
                
                for nama_model, model_key in models_to_run.items():
                    model_prediksi = target_data[model_key]
                    
                    # Buat DataFrame kosong khusus untuk fitur yang dibutuhkan
                    df_X = pd.DataFrame(columns=fitur_terpilih)
                    df_X.loc[0] = 0 # Inisialisasi baris pertama
                    
                    # Isi nilai dari mapped_input ke kolom yang sesuai (Robust matching)
                    for col in df_X.columns:
                        for key in mapped_input:
                            if col == key or col.startswith(key[:20]):
                                df_X.at[0, col] = mapped_input[key]
                                break
                    
                    # Lakukan Prediksi
                    pred_val = model_prediksi.predict(df_X)[0]
                    # Simpan hasil di dictionary menggunakan nama display yang bersih
                    hasil_prediksi[nama_display][nama_model] = np.clip(pred_val, 0, 100)
            
            st.session_state['hasil_prediksi'] = hasil_prediksi
            st.session_state['prediksi_selesai'] = True

    # --- 6. TAMPILKAN HASIL ---
    if st.session_state.get('prediksi_selesai', False):
        st.success("✅ Analisis Berhasil! Proyeksi nilai dari ketiga algoritma:")
        
        hasil = st.session_state['hasil_prediksi']
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.markdown("### 📐 Matematika")
            st.info(f"**MLR:** {hasil['Matematika']['MLR']:.2f}")
            st.success(f"**SVR:** {hasil['Matematika']['SVR']:.2f}")
            st.warning(f"**RFR:** {hasil['Matematika']['RFR']:.2f}")

        with col_m2:
            st.markdown("### 🗣️ B. Inggris")
            st.info(f"**MLR:** {hasil['Bahasa Inggris']['MLR']:.2f}")
            st.success(f"**SVR:** {hasil['Bahasa Inggris']['SVR']:.2f}")
            st.warning(f"**RFR:** {hasil['Bahasa Inggris']['RFR']:.2f}")

        with col_m3:
            st.markdown("### 📖 B. Indonesia")
            st.info(f"**MLR:** {hasil['Bahasa Indonesia']['MLR']:.2f}")
            st.success(f"**SVR:** {hasil['Bahasa Indonesia']['SVR']:.2f}")
            st.warning(f"**RFR:** {hasil['Bahasa Indonesia']['RFR']:.2f}")
        
        st.markdown("---")
        st.caption("🔍 **Catatan Analisis**: Karena ketiga model secara statistik setara (*p-value* > 0.05 pada Uji Wilcoxon), nilai-nilai di atas merupakan rentang wajar prediksi yang sah. Anda dapat menggunakan nilai rata-rata dari ketiga model ini sebagai prediksi final (Ensemble) yang sangat stabil.")