# app.py
import os
from io import BytesIO
import base64


from flask import Flask, render_template, request
from PIL import Image

import torch
from torchvision import transforms

from model import load_model

# --------- Config ---------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model at startup
MODEL_PATH = "skin_cancer_cnn.pth"
model, class_names = load_model(MODEL_PATH, device)

# Same transforms as training (adjust if needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   
        std=[0.229, 0.224, 0.225]
    )
])


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image: Image.Image):
    """Run model prediction on a single PIL image."""
    image = image.convert("RGB")
    img_t = transform(image)
    img_t = img_t.unsqueeze(0)  # [1, 3, H, W]

    with torch.no_grad():
        img_t = img_t.to(device)
        log_probs = model(img_t)               # output of log_softmax
        probs = log_probs.exp().cpu().numpy()[0]  # convert log-probs → probs

    predicted_idx = probs.argmax()
    predicted_class = class_names[predicted_idx]
    confidence = float(probs[predicted_idx])

    return predicted_class, confidence, probs



@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error_msg = None
    probs_percent = None
    image_data = None       # for preview

    if request.method == "POST":
        if "file" not in request.files:
            error_msg = "No file part in the request."
        else:
            file = request.files["file"]
            if file.filename == "":
                error_msg = "No file selected."
            elif not allowed_file(file.filename):
                error_msg = "File type not allowed. Please upload PNG, JPG or JPEG."
            else:
                try:
                    # read bytes
                    image_bytes = file.read()

                    # PIL image for model
                    image = Image.open(BytesIO(image_bytes))

                    # base64 for preview in the browser
                    preview_img = image.convert("RGB")
                    buf = BytesIO()
                    preview_img.save(buf, format="PNG")
                    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    image_data = f"data:image/png;base64,{img_b64}"

                    # run model
                    predicted_class, confidence, probs = predict_image(image)
                    prediction = predicted_class
                    probs_percent = [float(p) * 100 for p in probs]

                except Exception as e:
                    error_msg = f"Error processing image: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error_msg=error_msg,
        class_names=class_names,
        probs=probs_percent,
        image_data=image_data,
    )



if __name__ == "__main__":
    app.run(debug=True)
