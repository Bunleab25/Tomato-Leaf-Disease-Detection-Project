import os
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from predict_mobilenet import (
    DATASET_ROOT,
    DEFAULT_MODEL_PATH,
    load_model_and_classes,
    predict_tensor,
    preprocess_pil_image,
)
from predict_yolo import (
    DEFAULT_YOLO_WEIGHTS,
    crop_from_bbox,
    detect_leaves,
    detection_to_dict,
    load_yolo_model,
    select_largest_detection,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MOBILENET_MODEL_PATH", DEFAULT_MODEL_PATH))
DATASET_PATH = Path(os.getenv("DATASET_ROOT", DATASET_ROOT))
PREDICT_API_KEY = os.getenv("PREDICT_API_KEY", "").strip()
YOLO_WEIGHTS_PATH = Path(os.getenv("YOLO_WEIGHTS_PATH", str(DEFAULT_YOLO_WEIGHTS)))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

model = None
class_names: list[str] = []
device = None
yolo_model = None


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,X-API-Key"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


def is_api_key_valid() -> bool:
    if not PREDICT_API_KEY:
        return True
    client_key = request.headers.get("X-API-Key", "").strip()
    return client_key == PREDICT_API_KEY


def initialize_model() -> None:
    global model, class_names, device, yolo_model
    if model is None:
        model, class_names, device = load_model_and_classes(MODEL_PATH, DATASET_PATH)
    if yolo_model is None:
        yolo_model = load_yolo_model(YOLO_WEIGHTS_PATH)


def run_hybrid_prediction(image: Image.Image) -> dict:
    detections = detect_leaves(image=image, model=yolo_model, conf_threshold=YOLO_CONF)

    if not detections:
        return {
            "prediction": "No leaf detected",
            "confidence": 0.0,
            "probabilities": {},
            "top_predictions": [],
            "classes": class_names,
            "detections_count": 0,
            "total_detections": 0,
            "detections": [],
            "no_detection": True,
            "selection_policy": "largest_box",
        }

    selected_detection = select_largest_detection(detections)
    if selected_detection is None:
        return {
            "prediction": "No leaf detected",
            "confidence": 0.0,
            "probabilities": {},
            "top_predictions": [],
            "classes": class_names,
            "detections_count": 0,
            "total_detections": 0,
            "detections": [],
            "no_detection": True,
            "selection_policy": "largest_box",
        }

    crop = crop_from_bbox(image, selected_detection.bbox_xyxy)
    tensor = preprocess_pil_image(crop)
    cls_result = predict_tensor(
        image_tensor=tensor,
        model=model,
        class_names=class_names,
        device=device,
        top_k=len(class_names),
    )

    selected_entry = detection_to_dict(selected_detection)
    selected_entry["classification"] = {
        "prediction": cls_result["prediction"],
        "confidence": cls_result["confidence"],
        "probabilities": cls_result["probabilities"],
        "top_predictions": cls_result["top_predictions"],
    }

    return {
        "prediction": cls_result["prediction"],
        "confidence": cls_result["confidence"],
        "probabilities": cls_result["probabilities"],
        "top_predictions": cls_result["top_predictions"],
        "classes": class_names,
        "detections_count": 1,
        "total_detections": len(detections),
        "detections": [selected_entry],
        "no_detection": False,
        "selection_policy": "largest_box",
    }


def read_image_from_request() -> Image.Image:
    uploaded = request.files.get("file")

    if uploaded and uploaded.filename is not None:
        raw_bytes = uploaded.read()
    elif request.data:
        raw_bytes = request.data
    else:
        raise ValueError("No image payload found. Use multipart field name 'file'.")

    if not raw_bytes:
        raise ValueError("Image payload is empty.")

    try:
        return Image.open(BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid image format.") from exc


@app.route("/", methods=["GET"])
def home():
    return render_template("dashboard.html")


@app.route("/health", methods=["GET"])
def health():
    initialize_model()
    return jsonify(
        {
            "status": "ok",
            "model": str(MODEL_PATH),
            "yolo_model": str(YOLO_WEIGHTS_PATH),
            "dataset_root": str(DATASET_PATH),
            "classes": class_names,
            "device": str(device),
            "yolo_conf": YOLO_CONF,
            "api_key_required": bool(PREDICT_API_KEY),
        }
    )


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    if not is_api_key_valid():
        return jsonify({"error": "Unauthorized: invalid API key"}), 401

    initialize_model()

    try:
        image = read_image_from_request()
        result = run_hybrid_prediction(image)

        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    initialize_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
