
import streamlit as st, pandas as pd
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Resultados",page_icon="📊",layout="wide")
from config import MODELOS_DIR, MODELO_FINETUNED, MODELO

st.title("📊 Resultados e Métricas")
st.divider()

bp=MODELOS_DIR/"cam1_best.pt"
if not bp.exists(): st.warning("Sem modelo. Execute 🧠 Treinar."); st.stop()

tam=bp.stat().st_size/1024/1024
c1,c2=st.columns(2)
c1.metric("Modelo","cam1_best.pt")
c2.metric("Tamanho",f"{tam:.1f} MB")

csv=MODELO_FINETUNED/"cam1"/"results.csv"
if csv.exists():
    st.markdown("### 📈 Curvas de treinamento")
    try:
        import plotly.graph_objects as go
        df=pd.read_csv(csv); df.columns=df.columns.str.strip()
        t1,t2=st.tabs(["mAP e Loss","Precisão e Recall"])
        with t1:
            fig=go.Figure()
            if "metrics/mAP50(B)" in df.columns:
                fig.add_trace(go.Scatter(y=df["metrics/mAP50(B)"],name="mAP@50",
                              line=dict(color="#2E6DA4",width=2)))
            if "train/box_loss" in df.columns:
                fig.add_trace(go.Scatter(y=df["train/box_loss"],name="Loss",
                              line=dict(color="#A46D2E",dash="dash"),yaxis="y2"))
            fig.update_layout(xaxis_title="Época",yaxis_title="mAP",
                              yaxis2=dict(title="Loss",overlaying="y",side="right"),height=400)
            st.plotly_chart(fig,use_container_width=True)
        with t2:
            fig2=go.Figure()
            for col,cor in [("metrics/precision(B)","#2E6DA4"),("metrics/recall(B)","#1A5C3A")]:
                if col in df.columns:
                    fig2.add_trace(go.Scatter(y=df[col],name=col.split("/")[1].split("(")[0],
                                              line=dict(color=cor,width=2)))
            fig2.update_layout(xaxis_title="Época",height=400)
            st.plotly_chart(fig2,use_container_width=True)
    except Exception as e: st.error(f"Erro: {e}")
else:
    st.info("Métricas aparecerão após o treinamento.")

st.divider()
st.markdown("### 🖼️ Gráficos YOLO")
graficos={"Curva PR":MODELO_FINETUNED/"cam1"/"PR_curve.png",
           "Conf. Matrix":MODELO_FINETUNED/"cam1"/"confusion_matrix.png",
           "Batch Treino":MODELO_FINETUNED/"cam1"/"train_batch0.jpg"}
ex={k:v for k,v in graficos.items() if v.exists()}
if ex:
    cs=st.columns(min(len(ex),3))
    for i,(n,p) in enumerate(ex.items()): cs[i%3].image(Image.open(p),caption=n,use_column_width=True)
else: st.info("Gráficos aparecerão após o treinamento.")

st.divider()
st.markdown("### 📦 Exportar para produção")
fmts=st.multiselect("Formato",["onnx","tflite","torchscript"],default=["onnx"])
if st.button("📦 Exportar",type="primary",use_container_width=True):
    try:
        from ultralytics import YOLO
        m=YOLO(str(bp))
        for fmt in fmts:
            with st.spinner(f"Exportando {fmt}..."): m.export(format=fmt,imgsz=MODELO["imgsz"])
            st.success(f"✅ {fmt} exportado")
    except Exception as e: st.error(f"Erro: {e}")
