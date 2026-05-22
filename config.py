import os
from pathlib import Path

BASE_DIR     = Path(__file__).parent
DADOS_DIR    = BASE_DIR / "dados"
MODELOS_DIR  = BASE_DIR / "modelos"
RESULT_DIR   = BASE_DIR / "resultados"

for d in [
    DADOS_DIR / "bruto",
    DADOS_DIR / "processado" / "images" / "train",
    DADOS_DIR / "processado" / "images" / "val",
    DADOS_DIR / "processado" / "images" / "test",
    DADOS_DIR / "processado" / "labels" / "train",
    DADOS_DIR / "processado" / "labels" / "val",
    DADOS_DIR / "processado" / "labels" / "test",
    MODELOS_DIR / "finetuned",
    MODELOS_DIR / "exportados",
    RESULT_DIR / "frames_erro",
]:
    d.mkdir(parents=True, exist_ok=True)

PROC_DIR         = DADOS_DIR / "processado"
IMAGES_TRAIN     = PROC_DIR / "images" / "train"
IMAGES_VAL       = PROC_DIR / "images" / "val"
IMAGES_TEST      = PROC_DIR / "images" / "test"
LABELS_TRAIN     = PROC_DIR / "labels" / "train"
LABELS_VAL       = PROC_DIR / "labels" / "val"
LABELS_TEST      = PROC_DIR / "labels" / "test"
MODELO_FINETUNED = MODELOS_DIR / "finetuned"
MODELO_EXPORTADO = MODELOS_DIR / "exportados"

CAMERA = {
    "fonte": 0, "largura": 1280, "altura": 720,
    "roi": (0.10, 0.15, 0.90, 0.85), "iluminacao": "reflexo",
}

DATASET = {
    "split_treino": 0.70, "split_val": 0.20, "split_test": 0.10,
    "min_imgs_por_classe": 80,
    "aug": {
        "hsv_h": 0.01, "hsv_s": 0.40, "hsv_v": 0.40,
        "degrees": 20, "translate": 0.10, "scale": 0.50,
        "shear": 5, "flipud": 0.30, "fliplr": 0.50,
        "mosaic": 1.0, "mixup": 0.10, "copy_paste": 0.10,
    }
}

MODELO = {
    "base": "yolov8s.pt",
    "classes": ['peca'],
    "epochs": 50, "batch": -1, "imgsz": 640,
    "lr0": 0.01, "lrf": 0.01, "momentum": 0.937,
    "weight_decay": 0.0005, "warmup_epochs": 3,
    "patience": 20,
    "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    "save_period": 10, "conf_threshold": 0.60,
    "iou_threshold": 0.45, "conf_s2_threshold": 0.85,
}

OCR = {
    "motor": "easyocr", "idiomas": ["en"],
    "padrao_codigo": r"[A-Z]{2}-\d{4}-[A-Z]",
    "conf_min": 0.80,
    "pipelines": ["normal", "binarizado", "reflexo", "nitidez"],
}

EXPORT = {"formatos": ["onnx"], "simplify_onnx": True, "opset": 12, "half": False}
LOG    = {"salvar_frames_erro": True, "salvar_frames_ok": False, "nivel": "INFO"}
