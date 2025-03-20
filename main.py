from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import json
from tensorflow.keras.models import load_model
from models.evaluation import evaluate_single_image
from utils import set_japanese_font

app = FastAPI()

# Enable CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set the Japanese font (if required for your evaluation visuals)
set_japanese_font()

# Load the model and class names on startup
MODEL_PATH = "best_model.h5"
CLASS_NAMES_PATH = "class_names.json"

model = load_model(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the class of an uploaded image.
    The uploaded file is temporarily saved and passed to evaluate_single_image.
    """
    # Save the uploaded file to a temporary file
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Use your evaluation function to predict the class
        prediction = evaluate_single_image(model, tmp_path, class_names)
    except Exception as e:
        os.remove(tmp_path)
        return {"error": str(e)}
    
    # Clean up the temporary file
    os.remove(tmp_path)
    
    return {"predicted_class": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)