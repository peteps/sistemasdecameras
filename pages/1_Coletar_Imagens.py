
import streamlit as st, cv2, numpy as np, tempfile
from pathlib import Path
from PIL import Image
import io

st.set_page_config(page_title="Coletar Imagens", page_icon="📷", layout="wide")
from config import IMAGES_TRAIN, DATASET, CAMERA

st.title("📷 Etapa 1 — Coleta de Imagens")
st.divider()

min_rec  = DATASET["min_imgs_por_classe"]
imgs_now = len(list(IMAGES_TRAIN.glob("*.jpg"))+list(IMAGES_TRAIN.glob("*.png")))
c1,c2,c3 = st.columns(3)
c1.metric("Coletadas", imgs_now)
c2.metric("Mínimo", min_rec)
c3.metric("Status", "✅ OK" if imgs_now>=min_rec else "⚠️ Insuficiente")
st.progress(min(imgs_now/min_rec,1.0))
st.divider()

modo = st.radio("Fonte:", ["Upload de vídeo", "Upload de imagens"], horizontal=True)
st.divider()

if modo == "Upload de vídeo":
    vf = st.file_uploader("Vídeo do posto (mp4/avi/mov)", type=["mp4","avi","mov"])
    c_cfg, c_prev = st.columns([1,2])
    with c_cfg:
        intv   = st.slider("1 frame a cada N", 5, 60, 15)
        maxf   = st.number_input("Máximo de frames (0=sem limite)", 0, 2000, 200)
        roi_on = st.checkbox("Recortar ROI", True)
        roi = None
        if roi_on:
            x0=st.slider("ROI X início (%)",0,50,10)
            y0=st.slider("ROI Y início (%)",0,50,15)
            x1=st.slider("ROI X fim (%)",50,100,90)
            y1=st.slider("ROI Y fim (%)",50,100,85)
            roi=(x0/100,y0/100,x1/100,y1/100)
    with c_prev:
        if vf:
            with tempfile.NamedTemporaryFile(suffix=".mp4",delete=False) as t:
                t.write(vf.read()); tp=t.name
            cap=cv2.VideoCapture(tp)
            tf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps=cap.get(cv2.CAP_PROP_FPS) or 30
            ok,frame=cap.read(); cap.release()
            if ok:
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                if roi:
                    h,w=rgb.shape[:2]
                    cv2.rectangle(rgb,(int(roi[0]*w),int(roi[1]*h)),
                                  (int(roi[2]*w),int(roi[3]*h)),(0,200,100),3)
                st.image(rgb,use_column_width=True)
                st.info(f"{tf} frames @ {fps:.0f}fps → ~{min(tf//intv, maxf or 9999)} extraídos")
    if vf and st.button("▶️ Extrair frames", type="primary", use_container_width=True):
        IMAGES_TRAIN.mkdir(parents=True, exist_ok=True)
        bar=st.progress(0,"Extraindo...")
        cap=cv2.VideoCapture(tp)
        tf2=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        salvos=idx=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            if idx%intv==0:
                if roi:
                    h,w=frame.shape[:2]
                    frame=frame[int(roi[1]*h):int(roi[3]*h),int(roi[0]*w):int(roi[2]*w)]
                cv2.imwrite(str(IMAGES_TRAIN/f"cam1_{idx:07d}.jpg"),frame,
                            [cv2.IMWRITE_JPEG_QUALITY,95])
                salvos+=1
                bar.progress(min(idx/tf2,1.0),f"{salvos} frames salvos")
                if maxf and salvos>=maxf: break
            idx+=1
        cap.release()
        st.success(f"✅ {salvos} frames extraídos"); st.balloons()
else:
    imgs=st.file_uploader("Imagens (JPG/PNG)",type=["jpg","jpeg","png"],accept_multiple_files=True)
    dest=st.selectbox("Destino:",["Treino","Validação","Teste"])
    mapa={"Treino":IMAGES_TRAIN,
          "Validação":IMAGES_TRAIN.parent.parent/"images"/"val",
          "Teste":IMAGES_TRAIN.parent.parent/"images"/"test"}
    if imgs:
        cols=st.columns(min(len(imgs),6))
        for i,f in enumerate(imgs[:6]):
            cols[i].image(Image.open(io.BytesIO(f.read())),use_column_width=True); f.seek(0)
        if st.button("💾 Salvar",type="primary",use_container_width=True):
            p=mapa[dest]; p.mkdir(parents=True,exist_ok=True)
            for f in imgs: (p/f.name).write_bytes(f.read())
            st.success(f"✅ {len(imgs)} imagens salvas")

st.divider()
with st.expander("💡 Dicas"):
    st.markdown("- **Mínimo:** 80 imagens por classe\n"
                "- **Varie:** posição, ângulo, iluminação\n"
                "- **Evite:** frames borrados ou com reflexo total\n"
                "- **Próximo passo:** anotar no Roboflow → ⚙️ Pré-processar")
