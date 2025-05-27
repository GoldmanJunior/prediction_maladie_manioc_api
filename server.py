from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Charger le modèle
model = tf.keras.models.load_model('cassava_model.h5')
class_names = ['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy']
print("Modèle chargé :", model)  # Pour confirmer qu'il est bien chargé


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        class_name = class_names[class_idx]
        return JSONResponse({
            'class': class_name,
            'confidence': confidence
        })
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)
