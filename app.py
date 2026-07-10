import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
import plotly.express as px
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
# CSS kustom — tampilan lebih modern
# ==========================================================
st.markdown("""
<style>
    /* ---------- Global ---------- */
    .stApp {
        background: linear-gradient(180deg, #f4f9f4 0%, #eef6f0 100%);
    }

    /* ---------- Header ---------- */
    .main-header {
        background: linear-gradient(120deg, #11998e 0%, #38ef7d 100%);
        padding: 2.2rem 1.5rem;
        border-radius: 20px;
        margin-bottom: 1.8rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 24px rgba(17, 153, 142, 0.35);
        position: relative;
        overflow: hidden;
    }
    .main-header h1 {
        font-size: 2.1rem;
        margin-bottom: 0.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.95;
        margin: 0;
    }

    /* ---------- Cards umum ---------- */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 4px solid #11998e;
    }

    /* ---------- Kartu hasil prediksi ---------- */
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.8rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(118, 75, 162, 0.35);
        animation: fadeInUp 0.5s ease;
    }
    .prediction-result h2 {
        font-size: 1.9rem;
        margin: 0.2rem 0;
    }
    .confidence-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        margin-top: 0.5rem;
        backdrop-filter: blur(4px);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ---------- Tombol ---------- */
    .stButton > button {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 700;
        transition: all 0.25s ease;
        box-shadow: 0 4px 12px rgba(17,153,142,0.25);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(17,153,142,0.4);
    }

    /* ---------- Info & warning box ---------- */
    .info-box {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        padding: 1.1rem;
        border-radius: 14px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 12px rgba(17,153,142,0.25);
    }
    .warning-box {
        background: #fff8e1;
        padding: 1.1rem;
        border-radius: 14px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        color: #6b4e00;
    }

    /* ---------- Tabs (Upload vs Kamera) ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 0.4rem;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        border-radius: 10px;
        padding: 0 1.2rem;
        font-weight: 600;
        color: #2f4f4f;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #11998e, #38ef7d) !important;
        color: white !important;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f2f9f4 100%);
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
        background: white;
        padding: 0.55rem 0.8rem;
        border-radius: 10px;
        margin-bottom: 0.45rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        border-left: 4px solid var(--chip-color, #11998e);
        font-size: 0.92rem;
        font-weight: 600;
        color: #2f2f2f;
    }

    /* ---------- Upload area label ---------- */
    .section-title {
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 0.4rem;
    }

    /* ---------- Footer ---------- */
    .app-footer {
        text-align: center;
        color: #4b4b4b;
        padding: 1.4rem 1rem 0.6rem 1rem;
    }
    .app-footer p { margin: 0.15rem 0; }

    /* ==========================================================
       Optimasi tampilan Mobile
       ========================================================== */
    @media (max-width: 768px) {
        /* Kurangi padding utama Streamlit di layar kecil */
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1.2rem !important;
        }

        .main-header {
            padding: 1.4rem 1rem;
            border-radius: 14px;
            margin-bottom: 1.2rem;
        }
        .main-header h1 {
            font-size: 1.35rem;
            line-height: 1.3;
        }
        .main-header p {
            font-size: 0.85rem;
        }

        .prediction-result {
            padding: 1.2rem;
            border-radius: 16px;
        }
        .prediction-result h2 {
            font-size: 1.4rem;
        }
        .confidence-badge {
            font-size: 0.85rem;
            padding: 0.3rem 0.8rem;
        }

        .section-title {
            font-size: 1rem;
        }

        /* Tab lebih ringkas & bisa discroll horizontal */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            padding: 0.3rem;
            overflow-x: auto;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            padding: 0 0.8rem;
            font-size: 0.85rem;
            white-space: nowrap;
        }

        .waste-chip {
            font-size: 0.85rem;
            padding: 0.45rem 0.65rem;
        }

        .info-box, .warning-box {
            padding: 0.85rem;
            font-size: 0.9rem;
        }

        .stButton > button {
            width: 100%;
            padding: 0.65rem 1rem;
            font-size: 0.95rem;
        }

        /* Kolom Streamlit otomatis stack di mobile, rapikan jaraknya */
        div[data-testid="column"] {
            padding-bottom: 0.5rem;
        }

        /* Kamera & gambar penuh lebar, tidak overflow */
        .stCamera, img {
            border-radius: 12px;
        }

        h1, h2, h3 {
            word-wrap: break-word;
        }
    }

    @media (max-width: 480px) {
        .main-header h1 {
            font-size: 1.15rem;
        }
        .prediction-result h2 {
            font-size: 1.2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# Inisialisasi status sesi
# ==========================================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

GARBAGE_CLASSES = {
    'battery': {
        'name': 'Baterai',
        'color': '#5D5D5D',
        'disposal': 'Daur ulang di tempat pengumpulan yang ditentukan',
        'icon': '🔋'
    },
    'biological': {
        'name': 'Limbah Biologis',
        'color': '#6B8E23',
        'disposal': 'Buat kompos atau buang di tempat sampah organik',
        'icon': '🍎'
    },
    'brown-glass': {
        'name': 'Kaca Coklat',
        'color': '#8B4513',
        'disposal': 'Bilas dan pisahkan berdasarkan warna',
        'icon': '🍾'
    },
    'cardboard': {
        'name': 'Kardus',
        'color': '#DEB887',
        'disposal': 'Bersihkan dan pipihkan sebelum didaur ulang',
        'icon': '📦'
    },
    'clothes': {
        'name': 'Pakaian',
        'color': '#D2691E',
        'disposal': 'Sumbangkan jika masih bisa digunakan, jika tidak buang di tempat daur ulang tekstil',
        'icon': '👕'
    },
    'green-glass': {
        'name': 'Kaca Hijau',
        'color': '#228B22',
        'disposal': 'Bilas dan pisahkan berdasarkan warna',
        'icon': '🥂'
    },
    'metal': {
        'name': 'Logam',
        'color': '#696969',
        'disposal': 'Bersihkan dan pisahkan logam besi/non-besi',
        'icon': '🔩'
    },
    'paper': {
        'name': 'Kertas',
        'color': '#DAA520',
        'disposal': 'Jaga agar tetap kering dan bersih',
        'icon': '📄'
    },
    'plastic': {
        'name': 'Plastik',
        'color': '#FF6347',
        'disposal': 'Periksa kode daur ulang dan bersihkan',
        'icon': '🥤'
    },
    'shoes': {
        'name': 'Sepatu',
        'color': '#A0522D',
        'disposal': 'Sumbangkan jika masih bisa digunakan, jika tidak buang di tempat daur ulang tekstil/sepatu',
        'icon': '👟'
    },
    'trash': {
        'name': 'Sampah Umum',
        'color': '#2F4F4F',
        'disposal': 'Buang di tempat sampah umum',
        'icon': '🗑️'
    },
    'white-glass': {
        'name': 'Kaca Bening',
        'color': '#BDB76B',
        'disposal': 'Bilas dan pisahkan berdasarkan warna',
        'icon': '🥛'
    }
}

# ==========================================================
# Fungsi memuat model dengan caching
# ==========================================================
@st.cache_resource
def load_classification_model():
    """Muat model klasifikasi sampah yang telah dilatih"""
    try:
        model = load_model('best_model.keras')
        if hasattr(model, 'class_indices'):
            st.write(f"Indeks kelas model: {model.class_indices}")
        return model
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {str(e)}")
        st.info("Pastikan file model 'best_model.keras' berada di direktori yang sama dengan skrip ini.")
        return None


def preprocess_image(img, target_size=(224, 224)):
    """Praproses gambar untuk prediksi model"""
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
    """Prediksi kelas sampah dari gambar"""
    try:
        processed_img = preprocess_image(img)
        if processed_img is None:
            return None, None

        predictions = model.predict(processed_img, verbose=0)
        probabilities = predictions[0]

        class_names = list(GARBAGE_CLASSES.keys())

        if len(probabilities) != len(class_names):
            st.error(f"Ketidaksesuaian: Model memprediksi {len(probabilities)} kelas, tetapi {len(class_names)} kelas ditentukan")

            if hasattr(model, 'class_names'):
                actual_class_names = model.class_names
                st.write(f"Nama kelas model: {actual_class_names}")

            if len(probabilities) < len(class_names):
                class_names = class_names[:len(probabilities)]
                st.warning(f"Menggunakan {len(probabilities)} kelas pertama: {class_names}")
            else:
                while len(class_names) < len(probabilities):
                    class_names.append(f"kelas_{len(class_names)}")
                st.warning(f"Nama kelas diperluas: {class_names}")

        results = {}
        for i, class_name in enumerate(class_names):
            if i < len(probabilities):
                results[class_name] = float(probabilities[i])

        max_idx = np.argmax(probabilities)
        if max_idx < len(class_names):
            predicted_class = class_names[max_idx]
            confidence = float(probabilities[max_idx])
        else:
            st.error(f"Indeks prediksi {max_idx} di luar jangkauan untuk nama kelas")
            return None, None

        return predicted_class, results
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat prediksi: {str(e)}")
        st.write(f"Detail pengecualian: {type(e).__name__}: {str(e)}")
        return None, None


def create_confidence_chart(results):
    """Buat bagan Confidence untuk prediksi"""
    if not results:
        return None

    classes = []
    confidences = []
    colors = []

    for class_name, confidence in results.items():
        if class_name in GARBAGE_CLASSES:
            classes.append(GARBAGE_CLASSES[class_name]['name'])
            confidences.append(confidence * 100)
            colors.append(GARBAGE_CLASSES[class_name]['color'])
        else:
            classes.append(class_name.title())
            confidences.append(confidence * 100)
            colors.append('#888888')

    if not classes:
        return None

    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=confidences,
            marker_color=colors,
            text=[f'{conf:.1f}%' for conf in confidences],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Confidence Prediksi berdasarkan Kelas",
        xaxis_title="Kelas Sampah",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        template="plotly_white"
    )

    return fig


def add_to_history(image_data, predicted_class, confidence, results, source_label):
    """Tambahkan prediksi ke riwayat"""
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'predicted_class': predicted_class,
        'confidence': confidence,
        'results': results,
        'source': source_label,
    }
    st.session_state.prediction_history.append(history_entry)


def run_prediction_flow(model, image, source_label):
    """Jalankan alur analisis untuk sebuah gambar dan tampilkan hasilnya"""
    if st.button(f"🔍 Analisis Gambar ({source_label})", type="primary", key=f"analisis_{source_label}"):
        with st.spinner("Menganalisis gambar..."):
            predicted_class, results = predict_garbage_class(model, image)

            if predicted_class and results:
                confidence = results[predicted_class]
                class_info = GARBAGE_CLASSES[predicted_class]

                st.markdown(f"""
                <div class="prediction-result">
                    <h2>{class_info['icon']} {class_info['name']}</h2>
                    <span class="confidence-badge">Confidence: {confidence*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 💡 Rekomendasi Pembuangan")

                if predicted_class == 'trash':
                    st.markdown("""
                    <div class="warning-box">
                        <strong>⚠️ Sampah Umum</strong><br>
                        Item ini harus dibuang di tempat sampah umum.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>♻️ Barang Daur Ulang</strong><br>
                        {class_info['disposal']}
                    </div>
                    """, unsafe_allow_html=True)

                add_to_history(None, predicted_class, confidence, results, source_label)
                st.session_state.latest_results = results


def main():
    # ---------------- Header ----------------
    st.markdown("""
    <div class="main-header">
        <h1>♻️ Sistem Klasifikasi Sampah Deep Learning</h1>
        <p>Unggah gambar atau ambil foto langsung untuk mengklasifikasikan jenis sampah dan dapatkan rekomendasi daur ulang</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- Muat model ----------------
    model = load_classification_model()
    if model is None:
        st.stop()

    # ---------------- Sidebar ----------------
    with st.sidebar:
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
            if 'latest_results' in st.session_state:
                del st.session_state['latest_results']
            st.success("Riwayat telah dibersihkan!")

    # ---------------- Konten utama ----------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<p class="section-title">📸 Pilih Sumber Gambar</p>', unsafe_allow_html=True)

        tab_camera, tab_upload = st.tabs(["📷 Ambil Foto", "📤 Upload Gambar"])

        # ---- Tab Kamera ----
        with tab_camera:
            camera_image = st.camera_input("Ambil foto langsung dari kamera")
            if camera_image is not None:
                try:
                    image_bytes = camera_image.getvalue()
                    cam_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    st.image(cam_image, caption="Foto yang Diambil", use_column_width=True)
                    run_prediction_flow(model, cam_image, "Kamera")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat membaca gambar: {type(e).__name__} - {e}")
            else:
                st.info("Aktifkan kamera dan ambil foto sampah untuk memulai analisis.")

        # ---- Tab Upload ----
        with tab_upload:
            uploaded_file = st.file_uploader(
                "Pilih gambar dari perangkat...",
                type=['jpg', 'jpeg', 'png'],
                help="Unggah gambar sampah untuk diklasifikasikan"
            )
            if uploaded_file is not None:
                try:
                    image_bytes = uploaded_file.getvalue()
                    upload_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    st.image(upload_image, caption="Gambar yang Diunggah", use_column_width=True)
                    run_prediction_flow(model, upload_image, "Upload")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat membaca gambar: {type(e).__name__} - {e}")
            else:
                st.info("Silakan unggah gambar (.jpg, .jpeg, .png) untuk memulai analisis.")

    with col2:
        st.markdown('<p class="section-title">📊 Hasil Analisis</p>', unsafe_allow_html=True)

        if hasattr(st.session_state, 'latest_results'):
            fig = create_confidence_chart(st.session_state.latest_results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Unggah gambar atau ambil foto, lalu klik tombol 'Analisis Gambar' untuk melihat hasilnya.")

        if st.session_state.prediction_history:
            st.markdown("### 📈 Session tatistic")

            total_predictions = len(st.session_state.prediction_history)
            avg_confidence = np.mean([entry['confidence'] for entry in st.session_state.prediction_history])

            classes_predicted = [entry['predicted_class'] for entry in st.session_state.prediction_history]
            most_common_class = max(set(classes_predicted), key=classes_predicted.count) if classes_predicted else None

            st.metric("Total Prediksi", total_predictions)
            st.metric("Rata-rata Confidence", f"{avg_confidence*100:.1f}%")
            if most_common_class:
                st.metric("Kelas Paling Umum", GARBAGE_CLASSES[most_common_class]['name'])

    # ---------------- Riwayat Prediksi ----------------
    if st.session_state.prediction_history:
        st.markdown("### 📋 Riwayat Prediksi")

        history_data = []
        for entry in st.session_state.prediction_history[-10:]:
            history_data.append({
                'Waktu': entry['timestamp'],
                'Sumber': entry.get('source', '-'),
                'Ikon': GARBAGE_CLASSES[entry['predicted_class']]['icon'],
                'Kelas Diprediksi': GARBAGE_CLASSES[entry['predicted_class']]['name'],
                'Confidence': f"{entry['confidence']*100:.1f}%",
            })

        if history_data:
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------------- Footer ----------------
    st.markdown("---")
    st.markdown("""
    <div class="app-footer">
        <p>🌱 Bantu lindungi lingkungan dengan klasifikasi dan daur ulang sampah yang tepat</p>
        <p>Dibuat oleh Kelompok 6 Machine Learning | 2025</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()