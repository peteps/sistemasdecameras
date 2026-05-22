
import streamlit as st, cv2, numpy as np, re, tempfile, time
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Inferência",page_icon="🔍",layout="wide")
from config import MODELOS_DIR, MODELO, OCR

st.title("🔍 Etapa 4 — Inferência ao Vivo")
st.divider()

mp=MODELOS_DIR/"cam1_best.pt"
if not mp.exists():
    st.error("Modelo não encontrado. Execute 🧠 Treinar primeiro.")
    if st.checkbox("Testar com modelo base yolov8s.pt"): mp=Path("yolov8s.pt")
    else: st.stop()

@st.cache_resource(show_spinner="Carregando YOLO...")
def load_yolo(p):
    from ultralytics import YOLO; return YOLO(p)

@st.cache_resource(show_spinner="Carregando EasyOCR...")
def load_ocr(ids):
    import easyocr; return easyocr.Reader(ids,gpu=False,verbose=False)

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
PIPES={"normal":ppn,"binarizado":ppb,"reflexo":ppr,"nitidez":ppk}

def inferir(frame,yolo,ocr_r):
    d=frame.copy(); h,w=d.shape[:2]
    pat=re.compile(OCR["padrao_codigo"])
    res={"codigo":None,"cy":0.0,"co":0.0,"status":"erro","bbox":None}
    dets=yolo(frame,conf=MODELO["conf_threshold"],iou=MODELO["iou_threshold"],verbose=False)
    roi=frame
    if dets and dets[0].boxes and len(dets[0].boxes)>0:
        boxes=dets[0].boxes; idx=int(boxes.conf.argmax())
        cy=float(boxes.conf[idx]); x1,y1,x2,y2=[int(v) for v in boxes.xyxy[idx].tolist()]
        res.update({"cy":cy,"bbox":(x1,y1,x2,y2)})
        p2=10; roi=frame[max(0,y1-p2):min(h,y2+p2),max(0,x1-p2):min(w,x2+p2)]
        cv2.rectangle(d,(x1,y1),(x2,y2),(0,220,100),2)
        cv2.putText(d,f"peca {cy:.0%}",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,220,100),1)
    best={"codigo":None,"conf":0.0}
    for nome,fn in PIPES.items():
        try: ls=ocr_r.readtext(fn(roi),detail=1,paragraph=False)
        except: continue
        for _,txt,cf in ls:
            txt=txt.strip().upper(); m=pat.search(txt)
            if m and float(cf)>best["conf"]:
                best={"codigo":m.group(),"conf":float(cf)}
            if best["conf"]>=OCR["conf_min"]: break
        if best["conf"]>=OCR["conf_min"]: break
    if best["codigo"]:
        res.update({"codigo":best["codigo"],"co":best["conf"],
                    "status":"ok" if best["conf"]>=OCR["conf_min"] else "warn"})
        bx=res["bbox"]
        if bx:
            lbl=f"{best['codigo']}  {best['conf']:.0%}"
            (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
            cv2.rectangle(d,(bx[0],bx[3]+2),(bx[0]+tw+8,bx[3]+th+10),(255,180,0),-1)
            cv2.putText(d,lbl,(bx[0]+4,bx[3]+th+4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),1)
    cores={"ok":(80,200,80),"warn":(40,160,240),"erro":(60,60,220)}
    cv2.rectangle(d,(0,0),(w,34),cores[res["status"]],-1)
    cv2.putText(d,f"ID:{res['codigo'] or '?'} YOLO:{res['cy']:.0%} OCR:{res['co']:.0%} [{res['status'].upper()}]",
                (8,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),1)
    res["frame"]=d; return res

try:
    yolo=load_yolo(str(mp)); ocr=load_ocr(OCR["idiomas"])
except Exception as e: st.error(f"Erro: {e}"); st.stop()

modo=st.radio("Modo:",["📷 Imagem","🎬 Vídeo"],horizontal=True)
st.divider()

if modo=="📷 Imagem":
    upl=st.file_uploader("Imagem da peça",type=["jpg","jpeg","png"])
    if upl:
        pil=Image.open(upl)
        frame=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)
        with st.spinner("Processando..."):
            t0=time.time(); res=inferir(frame,yolo,ocr); t=time.time()-t0
        ci,cr=st.columns([2,1])
        ci.image(cv2.cvtColor(res["frame"],cv2.COLOR_BGR2RGB),use_column_width=True)
        with cr:
            sm={"ok":("✅ Confirmado","success"),"warn":("⚠️ S2","warning"),"erro":("❌ Erro","error")}
            lbl,tp2=sm[res["status"]]; getattr(st,tp2)(lbl)
            st.metric("Código",res["codigo"] or "—")
            st.metric("Conf. YOLO",f"{res['cy']:.1%}")
            st.metric("Conf. OCR",f"{res['co']:.1%}")
            st.metric("Tempo",f"{t*1000:.0f} ms")
            if res["status"]=="ok":
                st.json({"id_peca":res["codigo"],"of":"OF-XXXX",
                         "setor":"Inspeção (CQ)","metodo":"C1 automático",
                         "conf":round(res["co"],3)})
else:
    vf=st.file_uploader("Vídeo",type=["mp4","avi","mov"])
    if vf:
        with tempfile.NamedTemporaryFile(suffix=".mp4",delete=False) as t: t.write(vf.read()); tp3=t.name
        cap=cv2.VideoCapture(tp3)
        tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fn=st.slider("Frame",0,max(0,tot-1),0)
        cap.set(cv2.CAP_PROP_POS_FRAMES,fn); ok2,frame=cap.read(); cap.release()
        if ok2:
            co2,cr2=st.columns(2)
            co2.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),caption=f"Frame {fn}",use_column_width=True)
            with st.spinner(): res=inferir(frame,yolo,ocr)
            cr2.image(cv2.cvtColor(res["frame"],cv2.COLOR_BGR2RGB),caption="Resultado",use_column_width=True)
            c1,c2,c3=st.columns(3)
            c1.metric("Código",res["codigo"] or "—")
            c2.metric("OCR",f"{res['co']:.1%}")
            c3.metric("Status",{"ok":"✅","warn":"⚠️","erro":"❌"}[res["status"]])
