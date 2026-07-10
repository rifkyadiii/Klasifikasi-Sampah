import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
import plotly.graph_objects as go
import pandas as pd
import cv2
from datetime import datetime

# ==========================================================
# Konfigurasi halaman
# ==========================================================
st.set_page_config(
    page_title="Klasifikasi Sampah Deep Learning",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CSS kustom — tampilan lebih modern & Tab Seimbang
# ==========================================================
st.markdown("""
<style>
    :root {
        --app-card-bg: var(--secondary-background-color, #ffffff);
        --app-card-text: var(--text-color, #262626);
        --app-muted-text: var(--text-color, #4b4b4b);
        --app-border-soft: rgba(128, 128, 128, 0.25);
        --app-shadow: rgba(0, 0, 0, 0.08);
    }

    .stApp {
        background: var(--background-color, #f4f9f4);
    }

    /* ---------- Header ---------- */
    .main-header {
        background: linear-gradient(120deg, #11998e 0%, #38ef7d 100%);
        padding: 2.2rem 1.5rem;
        border-radius: 20px;
        margin-bottom: 1.8rem;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 8px 24px rgba(17, 153, 142, 0.35);
    }
    .main-header h1 {
        font-size: 2.1rem;
        margin-bottom: 0.3rem;
        font-weight: 800;
        color: #ffffff;
    }
    .main-header p {
        font-size: 1rem;
        margin: 0;
        color: #ffffff;
    }

    /* ---------- Kartu hasil prediksi ---------- */
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.8rem;
        border-radius: 20px;
        color: #ffffff;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(118, 75, 162, 0.35);
        animation: fadeInUp 0.5s ease;
    }
    .prediction-result h2 {
        font-size: 1.9rem;
        margin: 0.2rem 0;
        color: #ffffff;
    }
    .confidence-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        color: #ffffff;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        margin-top: 0.5rem;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ---------- Tombol ---------- */
    .stButton > button {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.2rem; /* Diperbesar sedikit agar lebih prominent */
        font-size: 1.1rem;
        font-weight: 700;
        transition: all 0.25s ease;
        box-shadow: 0 4px 12px rgba(17,153,142,0.25);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(17,153,142,0.4);
        color: #ffffff;
    }

    /* ---------- Info & warning box ---------- */
    .info-box {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        padding: 1.1rem;
        border-radius: 14px;
        margin: 1rem 0;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(17,153,142,0.25);
    }
    .warning-box {
        background: var(--app-card-bg);
        padding: 1.1rem;
        border-radius: 14px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        color: var(--app-card-text);
        box-shadow: 0 2px 10px var(--app-shadow);
    }

    /* ---------- Tabs (Full Width & Seimbang) ---------- */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        width: 100%;
        gap: 8px;
        background: var(--app-card-bg);
        padding: 0.4rem;
        border-radius: 14px;
        box-shadow: 0 2px 10px var(--app-shadow);
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1; /* Membuat ukuran setiap tab rata secara proporsional */
        height: 46px;
        border-radius: 10px;
        padding: 0 1rem;
        font-weight: 600;
        color: var(--app-card-text);
        display: flex;
        justify-content: center;
        align-items: center;
        white-space: nowrap;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #11998e, #38ef7d) !important;
        color: #ffffff !important;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: var(--background-color, #ffffff);
    }
    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #11998e;
        margin-bottom: 0.8rem;
    }
    .waste-chip {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        background: var(--app-card-bg);
        color: var(--app-card-text);
        padding: 0.55rem 0.8rem;
        border-radius: 10px;
        margin-bottom: 0.45rem;
        box-shadow: 0 1px 4px var(--app-shadow);
        border-left: 4px solid var(--chip-color, #11998e);
        font-weight: 600;
    }
    .stats-box {
        background: var(--app-card-bg);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 1px 4px var(--app-shadow);
        margin-bottom: 1rem;
        border-top: 3px solid #11998e;
    }
    .section-title {
        font-weight: 800;
        color: #11998e;
        margin-bottom: 0.4rem;
        font-size: 1.2rem;
    }
    .app-footer {
        text-align: center;
        color: var(--app-muted-text);
        padding: 1.4rem 1rem 0.6rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# Konfigurasi Kelas dan Status Sesi
# ==========================================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'latest_results' not in st.session_state:
    st.session_state.latest_results = None
if 'latest_prediction_info' not in st.session_state:
    st.session_state.latest_prediction_info = None

GARBAGE_CLASSES = {
    'battery': {'name': 'Baterai', 'color': '#5D5D5D', 'disposal': 'Daur ulang di tempat pengumpulan yang ditentukan', 'icon': '🔋'},
    'biological': {'name': 'Limbah Biologis', 'color': '#6B8E23', 'disposal': 'Buat kompos atau buang di tempat sampah organik', 'icon': '🍎'},
    'brown-glass': {'name': 'Kaca Coklat', 'color': '#8B4513', 'disposal': 'Bilas dan pisahkan berdasarkan warna', 'icon': '🍾'},
    'cardboard': {'name': 'Kardus', 'color': '#DEB887', 'disposal': 'Bersihkan dan pipihkan sebelum didaur ulang', 'icon': '📦'},
    'clothes': {'name': 'Pakaian', 'color': '#D2691E', 'disposal': 'Sumbangkan jika masih bisa digunakan, jika tidak buang di tempat daur ulang tekstil', 'icon': '👕'},
    'green-glass': {'name': 'Kaca Hijau', 'color': '#228B22', 'disposal': 'Bilas dan pisahkan berdasarkan warna', 'icon': '🥂'},
    'metal': {'name': 'Logam', 'color': '#696969', 'disposal': 'Bersihkan dan pisahkan logam besi/non-besi', 'icon': '🔩'},
    'paper': {'name': 'Kertas', 'color': '#DAA520', 'disposal': 'Jaga agar tetap kering dan bersih', 'icon': '📄'},
    'plastic': {'name': 'Plastik', 'color': '#FF6347', 'disposal': 'Periksa kode daur ulang dan bersihkan', 'icon': '🥤'},
    'shoes': {'name': 'Sepatu', 'color': '#A0522D', 'disposal': 'Sumbangkan jika masih bisa digunakan, buang di tempat daur ulang tekstil/sepatu', 'icon': '👟'},
    'trash': {'name': 'Sampah Umum', 'color': '#2F4F4F', 'disposal': 'Buang di tempat sampah umum', 'icon': '🗑️'},
    'white-glass': {'name': 'Kaca Bening', 'color': '#BDB76B', 'disposal': 'Bilas dan pisahkan berdasarkan warna', 'icon': '🥛'}
}

# ==========================================================
# Fungsi Model dan Prediksi
# ==========================================================
@st.cache_resource
def load_classification_model():
    try:
        model = load_model('best_model.keras')
        return model
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {str(e)}")
        return None

def preprocess_image(img, target_size=(224, 224)):
    try:
        if isinstance(img, Image.Image):
            img_array = np.array(img.convert('RGB'))
        else:
            img_array = img
        img_resized = cv2.resize(img_array, target_size)
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch
    except Exception as e:
        st.error(f"Terjadi kesalahan saat praproses gambar: {str(e)}")
        return None

def predict_garbage_class(model, img):
    try:
        processed_img = preprocess_image(img)
        if processed_img is None:
            return None, None
        predictions = model.predict(processed_img, verbose=0)
        probabilities = predictions[0]
        class_names = list(GARBAGE_CLASSES.keys())
        
        results = {}
        for i, class_name in enumerate(class_names):
            if i < len(probabilities):
                results[class_name] = float(probabilities[i])
                
        max_idx = np.argmax(probabilities)
        predicted_class = class_names[max_idx]
        confidence = float(probabilities[max_idx])
        return predicted_class, results
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat prediksi: {str(e)}")
        return None, None

def create_confidence_chart(results):
    if not results:
        return None
    classes = [GARBAGE_CLASSES[c]['name'] if c in GARBAGE_CLASSES else c.title() for c in results.keys()]
    confidences = [c * 100 for c in results.values()]
    colors = [GARBAGE_CLASSES[c]['color'] if c in GARBAGE_CLASSES else '#888888' for c in results.keys()]

    fig = go.Figure(data=[
        go.Bar(
            x=classes, y=confidences, marker_color=colors,
            text=[f'{conf:.1f}%' for conf in confidences], textposition='auto',
        )
    ])
    fig.update_layout(
        title="",
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Kelas Sampah", yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]), height=350,
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def process_and_save_prediction(model, image, source_label):
    with st.spinner("Menganalisis gambar..."):
        predicted_class, results = predict_garbage_class(model, image)
        if predicted_class and results:
            confidence = results[predicted_class]
            
            st.session_state.prediction_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'source': source_label,
            })
            
            st.session_state.latest_results = results
            st.session_state.latest_prediction_info = {
                'class': predicted_class,
                'confidence': confidence,
                'info': GARBAGE_CLASSES[predicted_class]
            }
            
            # MEMAKSA HALAMAN REFRESH AGAR STATISTIK SIDEBAR LANGSUNG BERUBAH (REALTIME)
            st.experimental_rerun()

# ==========================================================
# Main App
# ==========================================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>♻️ Sistem Klasifikasi Sampah Deep Learning</h1>
        <p>Unggah gambar atau ambil foto langsung untuk mengklasifikasikan jenis sampah dan dapatkan rekomendasi daur ulang</p>
    </div>
    """, unsafe_allow_html=True)

    model = load_classification_model()
    if model is None:
        st.stop()

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown('<div class="sidebar-title">📊 Statistik Sesi</div>', unsafe_allow_html=True)
        if st.session_state.prediction_history:
            total_preds = len(st.session_state.prediction_history)
            avg_conf = np.mean([entry['confidence'] for entry in st.session_state.prediction_history])
            classes_pred = [entry['predicted_class'] for entry in st.session_state.prediction_history]
            common_class = max(set(classes_pred), key=classes_pred.count)
            common_name = GARBAGE_CLASSES[common_class]['name']
            
            st.markdown(f"""
            <div class="stats-box">
                <p style="margin:0; font-size:0.9rem;">Total Prediksi</p>
                <b style="font-size:1.4rem; color:#11998e;">{total_preds}</b>
                <hr style="margin:0.5rem 0;">
                <p style="margin:0; font-size:0.9rem;">Rata-rata Confidence</p>
                <b style="font-size:1.4rem; color:#11998e;">{avg_conf*100:.1f}%</b>
                <hr style="margin:0.5rem 0;">
                <p style="margin:0; font-size:0.9rem;">Kelas Terbanyak</p>
                <b style="font-size:1.1rem; color:#11998e;">{common_name}</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Belum ada data prediksi di sesi ini.")

        st.markdown('<div class="sidebar-title">♻️ Jenis Sampah</div>', unsafe_allow_html=True)
        for class_key, class_info in GARBAGE_CLASSES.items():
            st.markdown(f"""
            <div class="waste-chip" style="--chip-color:{class_info['color']}">
                <span style="font-size:1.2rem;">{class_info['icon']}</span> {class_info['name']}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑️ Bersihkan Riwayat"):
            st.session_state.prediction_history = []
            st.session_state.latest_results = None
            st.session_state.latest_prediction_info = None
            st.experimental_rerun()

    # ---------------- Layout Utama ----------------
    col1, col2 = st.columns([2, 1])

    # ---- Kolom 1: Sumber Gambar & Tombol Analisis ----
    with col1:
        st.markdown('<p class="section-title">📸 Sumber Gambar & Riwayat</p>', unsafe_allow_html=True)
        
        # Placeholder untuk menempatkan tombol di atas tab (selalu muncul)
        action_placeholder = st.empty()
        
        tab_camera, tab_upload, tab_history = st.tabs(["📷 Ambil Foto", "📤 Upload Gambar", "📋 Riwayat Prediksi"])
        
        img_to_process = None
        img_source = ""

        with tab_camera:
            camera_image = st.camera_input("Ambil foto langsung dari kamera")
            if camera_image is not None:
                try:
                    cam_img = Image.open(BytesIO(camera_image.getvalue())).convert("RGB")
                    
                    # HAPUS ATAU COMMENT BARIS DI BAWAH INI:
                    # st.image(cam_img, caption="Foto yang Diambil", use_column_width=True) 
                    
                    img_to_process = cam_img
                    img_source = "Kamera"
                except Exception as e:
                    st.error(f"Error: {e}")

        with tab_upload:
            uploaded_file = st.file_uploader("Pilih gambar dari perangkat...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                try:
                    up_img = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
                    st.image(up_img, caption="Gambar yang Diunggah", use_column_width=True)
                    # Upload akan meng-override kamera jika keduanya aktif
                    img_to_process = up_img
                    img_source = "Upload"
                except Exception as e:
                    st.error(f"Error: {e}")
                    
        with tab_history:
            if st.session_state.prediction_history:
                history_data = [{
                    'Waktu': entry['timestamp'],
                    'Sumber': entry['source'],
                    'Kelas': GARBAGE_CLASSES[entry['predicted_class']]['name'],
                    'Confidence': f"{entry['confidence']*100:.1f}%",
                } for entry in reversed(st.session_state.prediction_history[-10:])]
                st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)
            else:
                st.info("Riwayat prediksi masih kosong.")
                
        # Mengisi placeholder tombol di bagian atas
        with action_placeholder:
            if st.button("🔍 Analisis Gambar", type="primary", use_container_width=True):
                if img_to_process is not None:
                    process_and_save_prediction(model, img_to_process, img_source)
                else:
                    st.warning("Silakan ambil foto atau unggah gambar terlebih dahulu!")

    # ---- Kolom 2: Hasil, Rekomendasi, & Grafik Confidence ----
    with col2:
        st.markdown('<p class="section-title">💡 Hasil & Rekomendasi</p>', unsafe_allow_html=True)
        
        if st.session_state.latest_prediction_info:
            info = st.session_state.latest_prediction_info
            
            st.markdown(f"""
            <div class="prediction-result" style="margin-top:0;">
                <h2>{info['info']['icon']} {info['info']['name']}</h2>
                <span class="confidence-badge">Confidence: {info['confidence']*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

            if info['class'] == 'trash':
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ Sampah Umum</strong><br>
                    Item ini harus dibuang di tempat sampah umum.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info-box">
                    <strong>♻️ Cara Pembuangan</strong><br>
                    {info['info']['disposal']}
                </div>
                """, unsafe_allow_html=True)
                
            # GRAFIK CONFIDENCE SCORE (Pindah ke bawah hasil rekomendasi)
            st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
            st.markdown('<p class="section-title">📊 Grafik Confidence Score</p>', unsafe_allow_html=True)
            
            if st.session_state.latest_results:
                fig = create_confidence_chart(st.session_state.latest_results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Silakan unggah gambar atau ambil foto, lalu klik 'Analisis Gambar' untuk melihat hasil dan rekomendasi di sini.")

    # ---------------- Footer ----------------
    st.markdown("---")
    st.markdown("""
    <div class="app-footer">
        <p>🌱 Bantu lindungi lingkungan dengan klasifikasi dan daur ulang sampah yang tepat</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()