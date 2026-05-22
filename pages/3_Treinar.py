
import streamlit as st, yaml, shutil, pandas as pd
from pathlib import Path

st.set_page_config(page_title="Treinar", page_icon="🧠", layout="wide")
from config import PROC_DIR, MODELOS_DIR, MODELO_FINETUNED, MODELO, DATASET, IMAGES_TRAIN, IMAGES_VAL

st.title("🧠 Etapa 3 — Fine-tuning YOLOv8")
st.divider()

it=list(IMAGES_TRAIN.glob("*.jpg"))+list(IMAGES_TRAIN.glob("*.png"))
lt=list((PROC_DIR/"labels"/"train").glob("*.txt"))
c1,c2,c3,c4=st.columns(4)
c1.metric("Imgs treino",len(it),delta="OK" if len(it)>=80 else "⚠️")
c2.metric("Imgs val",len(list(IMAGES_VAL.glob("*.jpg"))),delta="OK")
c3.metric("Labels treino",len(lt),delta="OK" if len(lt)>=1 else "❌ anotar")
c4.metric("Modelo",("✅ Existe" if (MODELOS_DIR/"cam1_best.pt").exists() else "❌ Ausente"))

if len(lt)==0:
    st.error("**❌ Labels não encontrados!**\n\n"
             "Anote as imagens no [Roboflow](https://roboflow.com), "
             "exporte em **YOLO v8** e coloque os `.txt` em `dados/processado/labels/train/`")

st.divider()
c_cfg,c_inf=st.columns(2)
with c_cfg:
    base=st.selectbox("Modelo base",["yolov8n.pt","yolov8s.pt","yolov8m.pt"],index=1)
    epochs=st.slider("Épocas",10,200,MODELO["epochs"])
    dev=st.selectbox("Dispositivo",["cpu","0 (GPU)","mps"])
    dval="cpu" if dev=="cpu" else ("0" if "GPU" in dev else "mps")
    imgsz=st.select_slider("Resolução",[320,416,512,640],value=640)
    cls_in=st.text_input("Classes",",".join(MODELO["classes"]))
    classes=[c.strip() for c in cls_in.split(",")]
with c_inf:
    st.info(f"**Fine-tuning = Transfer Learning**\n\n"
            f"Parte do `{base}` pré-treinado no COCO e especializa para suas peças.\n\n"
            f"Estimativa ({epochs} épocas):\n- CPU: ~{epochs*45//60} min\n- GPU: ~{epochs*3//60} min")

def gerar_yaml():
    cfg={"path":str(PROC_DIR),"train":"images/train","val":"images/val",
         "test":"images/test","nc":len(classes),"names":classes}
    yp=PROC_DIR/"dataset.yaml"
    with open(yp,"w") as f: yaml.dump(cfg,f,default_flow_style=False)
    return yp

if st.button("🚀 Iniciar fine-tuning",type="primary",use_container_width=True,disabled=(len(lt)==0)):
    yp=gerar_yaml()
    try:
        from ultralytics import YOLO
        m=YOLO(base)
        bar=st.progress(0,"Iniciando..."); met_ph=st.empty(); hist=[]
        class CB:
            def on_train_epoch_end(self,t):
                ep=t.epoch+1; tot=t.epochs
                bar.progress(ep/tot,f"Época {ep}/{tot}")
                if hasattr(t,"metrics"):
                    mv=t.metrics
                    hist.append({"Época":ep,
                        "mAP50":round(mv.get("metrics/mAP50(B)",0),4),
                        "Precisão":round(mv.get("metrics/precision(B)",0),4),
                        "Recall":round(mv.get("metrics/recall(B)",0),4)})
                    with met_ph.container():
                        cc1,cc2,cc3=st.columns(3)
                        cc1.metric("mAP@50",f"{mv.get('metrics/mAP50(B)',0):.4f}")
                        cc2.metric("Precisão",f"{mv.get('metrics/precision(B)',0):.4f}")
                        cc3.metric("Recall",f"{mv.get('metrics/recall(B)',0):.4f}")
                        if len(hist)>1:
                            st.line_chart(pd.DataFrame(hist).set_index("Época"))
        cb=CB(); aug=DATASET["aug"]
        m.train(data=str(yp),epochs=epochs,imgsz=imgsz,device=dval,
                project=str(MODELO_FINETUNED),name="cam1",exist_ok=True,
                val=True,plots=True,verbose=False,
                hsv_h=aug["hsv_h"],hsv_s=aug["hsv_s"],hsv_v=aug["hsv_v"],
                degrees=aug["degrees"],flipud=aug["flipud"],fliplr=aug["fliplr"],
                mosaic=aug["mosaic"],mixup=aug["mixup"],
                callbacks={"on_train_epoch_end":[cb.on_train_epoch_end]})
        bs=MODELO_FINETUNED/"cam1"/"weights"/"best.pt"
        bd=MODELOS_DIR/"cam1_best.pt"
        if bs.exists(): shutil.copy(bs,bd)
        bar.progress(1.0,"✅ Concluído!"); st.success(f"Modelo: {bd}"); st.balloons()
        res=m.val(data=str(yp),verbose=False)
        cc1,cc2,cc3,cc4=st.columns(4)
        cc1.metric("mAP@50",f"{res.box.map50:.4f}")
        cc2.metric("mAP@50-95",f"{res.box.map:.4f}")
        cc3.metric("Precisão",f"{res.box.mp:.4f}")
        cc4.metric("Recall",f"{res.box.mr:.4f}")
        if res.box.map50>=0.90: st.success("🏆 Excelente! Pronto para produção.")
        elif res.box.map50>=0.75: st.warning("✅ Bom. Considere mais dados.")
        else: st.error("❌ Revisar dataset.")
    except ImportError: st.error("pip install ultralytics")
    except Exception as e: st.error(f"Erro: {e}")
