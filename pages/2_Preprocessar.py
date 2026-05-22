
import streamlit as st, cv2, numpy as np, random
from pathlib import Path
from PIL import Image
import io, pandas as pd

st.set_page_config(page_title="Pré-processar", page_icon="⚙️", layout="wide")
from config import IMAGES_TRAIN, DATASET

st.title("⚙️ Etapa 2 — Pré-processamento")
st.divider()

def ppn(i):
    g=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
    c=cv2.createCLAHE(2.0,(8,8)).apply(g)
    b=cv2.GaussianBlur(c,(0,0),3)
    return cv2.addWeighted(c,1.5,b,-0.5,0)
def ppb(i): return cv2.adaptiveThreshold(ppn(i),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,10)
def ppk(i): return cv2.filter2D(cv2.cvtColor(i,cv2.COLOR_BGR2GRAY),-1,np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32))
def ppr(i):
    h=cv2.cvtColor(i,cv2.COLOR_BGR2HSV)
    m=cv2.dilate(cv2.inRange(h,(0,0,220),(180,30,255)),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=2)
    r=i.copy(); r[m>0]=(120,120,120)
    return cv2.createCLAHE(2.5,(8,8)).apply(cv2.cvtColor(r,cv2.COLOR_BGR2GRAY))
PIPES={"Normal":ppn,"Binarizado":ppb,"Nitidez":ppk,"Anti-reflexo":ppr}

st.markdown("### 🔬 Comparar pipelines")
upl=st.file_uploader("Imagem de exemplo",type=["jpg","jpeg","png"])
if upl:
    pil=Image.open(upl)
    img_np=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)
    cols=st.columns(5)
    cols[0].image(pil,caption="Original",use_column_width=True)
    for i,(nome,fn) in enumerate(PIPES.items()):
        r=fn(img_np)
        if len(r.shape)==2: r=cv2.cvtColor(r,cv2.COLOR_GRAY2RGB)
        else: r=cv2.cvtColor(r,cv2.COLOR_BGR2RGB)
        cols[i+1].image(r,caption=nome,use_column_width=True)
    b=cv2.Laplacian(cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
    br=cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY).mean()
    rf=(cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY)>240).mean()
    c1,c2,c3=st.columns(3)
    c1.metric("Nitidez",f"{b:.1f}",delta="OK" if b>=80 else "⚠️ borrada")
    c2.metric("Brilho",f"{br:.1f}",delta="OK" if 40<=br<=210 else "⚠️ fora")
    c3.metric("Reflexo",f"{rf:.1%}",delta="OK" if rf<=0.15 else "⚠️ alto")

st.divider()
st.markdown("### ✅ Validar dataset")
if st.button("🔍 Analisar imagens de treino",use_container_width=True):
    imgs=list(IMAGES_TRAIN.glob("*.jpg"))+list(IMAGES_TRAIN.glob("*.png"))
    if not imgs: st.warning("Sem imagens. Vá para 📷 Coletar Imagens.")
    else:
        dados=[]; bar=st.progress(0)
        for i,p in enumerate(imgs):
            f=cv2.imread(str(p))
            if f is None: continue
            b=cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
            br=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).mean()
            rf=(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)>240).mean()
            prob=[]
            if b<80: prob.append("borrada")
            if br<40: prob.append("escura")
            if br>210: prob.append("clara")
            if rf>.15: prob.append("reflexo")
            dados.append({"Arquivo":p.name,"Nitidez":round(b,1),
                          "Brilho":round(br,1),"Status":"✅" if not prob else "⚠️ "+",".join(prob)})
            bar.progress((i+1)/len(imgs))
        df=pd.DataFrame(dados)
        nok=(df["Status"]=="✅").sum()
        st.metric("OK",nok); st.metric("Problemas",len(df)-nok)
        st.dataframe(df,use_container_width=True)

st.divider()
st.markdown("### 🔄 Augmentação offline")
imgs_now=len(list(IMAGES_TRAIN.glob("*.jpg"))+list(IMAGES_TRAIN.glob("*.png")))
fator=st.slider("Variações por imagem",1,6,3)
st.info(f"Resultado estimado: {imgs_now*(fator+1)} imagens")
if st.button("🔄 Gerar augmentações",type="primary",use_container_width=True):
    imgs=[f for f in list(IMAGES_TRAIN.glob("*.jpg")) if "_aug" not in f.stem]
    if not imgs: st.error("Sem imagens.")
    else:
        bar=st.progress(0); ger=0
        for i,ip in enumerate(imgs):
            fr=cv2.imread(str(ip))
            if fr is None: continue
            h,w=fr.shape[:2]; vs=[]
            for a in [-15,-7,7,15]:
                M=cv2.getRotationMatrix2D((w//2,h//2),a,1.0)
                vs.append(cv2.warpAffine(fr,M,(w,h),borderMode=cv2.BORDER_REFLECT_101))
            vs.append(cv2.flip(fr,1)); vs.append(cv2.flip(fr,0))
            for al,be in [(0.8,-30),(1.2,30)]:
                vs.append(cv2.convertScaleAbs(fr,alpha=al,beta=be))
            sel=random.sample(vs,min(fator,len(vs)))
            lb=ip.with_suffix(".txt")
            for j,v in enumerate(sel):
                nn=IMAGES_TRAIN/f"{ip.stem}_aug{j:02d}.jpg"
                cv2.imwrite(str(nn),v,[cv2.IMWRITE_JPEG_QUALITY,92])
                if lb.exists():
                    (IMAGES_TRAIN.parent.parent/"labels"/"train"/
                     f"{ip.stem}_aug{j:02d}.txt").write_text(lb.read_text())
                ger+=1
            bar.progress((i+1)/len(imgs))
        st.success(f"✅ {ger} imagens geradas"); st.balloons()
