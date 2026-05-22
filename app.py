
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Câmera 1 — Rastreabilidade Industrial",
    page_icon="🏭", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
section[data-testid="stSidebar"] { background-color: #1A3A5C !important; }
section[data-testid="stSidebar"] * { color: white !important; }
.card { background:white; border-radius:10px; padding:16px 20px;
        border:1px solid #D0D7E2; box-shadow:0 1px 3px rgba(0,0,0,.06); }
.step { border-left:4px solid #2E6DA4; padding:10px 16px;
        background:#F4F6F9; border-radius:0 8px 8px 0; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🏭 Câmera 1")
    st.caption("Rastreabilidade industrial")
    st.divider()
    st.page_link("app.py",                     label="🏠 Início")
    st.page_link("pages/1_Coletar_Imagens.py", label="📷 Coletar imagens")
    st.page_link("pages/2_Preprocessar.py",    label="⚙️ Pré-processar")
    st.page_link("pages/3_Treinar.py",         label="🧠 Treinar modelo")
    st.page_link("pages/4_Inferir.py",         label="🔍 Inferência")
    st.page_link("pages/5_Resultados.py",      label="📊 Resultados")
    st.divider()
    st.caption("PET Eng. de Produção · 2025")

st.title("🏭 Sistema de Rastreabilidade — Câmera 1")
st.subheader("Identificação automática de código de peças por visão computacional")
st.divider()

c1,c2,c3,c4 = st.columns(4)
for col, titulo, cor, desc in [
    (c1,"1. Coletar","#0F6E56","Frames do vídeo"),
    (c2,"2. Pré-processar","#2E6DA4","Validar e augmentar"),
    (c3,"3. Treinar","#A46D2E","Fine-tuning YOLOv8"),
    (c4,"4. Inferir","#1A5C3A","YOLO + OCR → ERP"),
]:
    col.markdown(
        f'<div class="card" style="border-top:4px solid {cor}">'
        f'<b style="color:{cor}">{titulo}</b><br>{desc}</div>',
        unsafe_allow_html=True
    )

st.divider()
st.markdown("### 🔄 Pipeline")
for txt in [
    "<b>Passo 1</b> — Frame capturado pela câmera do posto",
    "<b>Passo 2</b> — YOLO detecta a região da peça (bounding box)",
    "<b>Passo 3</b> — OCR testa 4 pipelines de pré-processamento",
    "<b>Passo 4</b> — Regex valida o padrão do código",
    "<b>Passo 5A ✅</b> — Confiança ≥ 85%: apontamento automático no ERP",
    "<b>Passo 5B ⚠️</b> — Confiança &lt; 85%: aciona S2 (seleção manual no tablet)",
]:
    st.markdown(f'<div class="step">{txt}</div>', unsafe_allow_html=True)

st.divider()
from config import MODELOS_DIR, IMAGES_TRAIN, IMAGES_VAL
c1,c2,c3 = st.columns(3)
imgs_t = len(list(IMAGES_TRAIN.glob("*.jpg"))+list(IMAGES_TRAIN.glob("*.png")))
imgs_v = len(list(IMAGES_VAL.glob("*.jpg"))+list(IMAGES_VAL.glob("*.png")))
c1.metric("Imagens de treino",    imgs_t, delta="OK" if imgs_t>=80 else "mín: 80")
c2.metric("Imagens de validação", imgs_v, delta="OK" if imgs_v>=20 else "mín: 20")
c3.metric("Modelo treinado",
          "✅ Sim" if (MODELOS_DIR/"cam1_best.pt").exists() else "❌ Não")
