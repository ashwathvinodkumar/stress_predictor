import os
import json
import threading
import time
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, db

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# --- Firebase Setup ---
firebase_key_json = os.environ["FIREBASE_KEY_JSON"]
firebase_cred_dict = json.loads(firebase_key_json)
cred = credentials.Certificate(firebase_cred_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://agri-hub-544be-default-rtdb.firebaseio.com'
})

# --- Load Model Artifacts ---
MODEL_PATH = "tamil_nadu_stress_model.pkl"
artifacts = joblib.load(MODEL_PATH)
model = artifacts['model']
scaler = artifacts['scaler']
encoders = artifacts['encoders']
le_stress = artifacts['le_stress']
feature_columns = artifacts['feature_columns']

# --- FastAPI App ---
app = FastAPI()

# --- Pydantic Model for API ---
class SensorData(BaseModel):
    mq135Voltage: float
    humidity: float
    soilMoisture: float
    temperature: float

# --- Prediction Function ---
def predict_stress_from_sensor(data: SensorData):
    try:
        voc = data.mq135Voltage
        temp = data.temperature
        humidity = data.humidity
        soil_moisture = data.soilMoisture
        crop = "Paddy"
        district = "Madurai"

        crop_enc = encoders['le_crop'].transform([crop])[0]
        district_enc = encoders['le_district'].transform([district])[0]
        soil_temp_interaction = soil_moisture * temp
        humidity_voc_interaction = humidity * voc
        heat_stress = 1 if (temp > 35 and humidity < 50) else 0
        drought_stress = 1 if (soil_moisture < 30) else 0

        data_dict = {
            'VOC_Volts': [voc],
            'Temperature_C': [temp],
            'Humidity_%': [humidity],
            'Soil_Moisture_%': [soil_moisture],
            'crop_encoded': [crop_enc],
            'district_encoded': [district_enc],
            'soil_temp_interaction': [soil_temp_interaction],
            'humidity_voc_interaction': [humidity_voc_interaction],
            'heat_stress': [heat_stress],
            'drought_stress': [drought_stress]
        }
        df = pd.DataFrame(data_dict)
        features_scaled = scaler.transform(df)
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0][prediction]
        stress_label = le_stress.inverse_transform([prediction])[0]

        # Update Firebase with prediction
        timestamp = datetime.now().isoformat()
        db.reference('sensorData/stress_prediction').set({
            "stress_type": stress_label,
            "confidence": float(round(confidence * 100, 2)),
            "timestamp": timestamp
        })
        print(f"✅ Stress Prediction: {stress_label}, Confidence: {confidence:.2%}, Time: {timestamp}")

        return {
            "stress_type": stress_label,
            "confidence": float(round(confidence * 100, 2)),
            "timestamp": timestamp
        }
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return {"error": str(e)}

# --- API Endpoint ---
@app.post("/predict")
def predict_route(data: SensorData):
    return predict_stress_from_sensor(data)

# --- Background Sensor Monitor ---
def monitor_firebase_sensor_data():
    last_values = None
    while True:
        try:
            ref = db.reference("sensorData")
            current = ref.get()
            if current is not None:
                if last_values is None or current != last_values:
                    print("🔔 Detected change in sensor data!")
                    required_fields = ['mq135Voltage', 'humidity', 'soilMoisture', 'temperature']
                    if all(field in current for field in required_fields):
                        data = SensorData(
                            mq135Voltage=float(current.get("mq135Voltage", 0.0)),
                            humidity=float(current.get("humidity", 0.0)),
                            soilMoisture=float(current.get("soilMoisture", 0.0)),
                            temperature=float(current.get("temperature", 0.0))
                        )
                        result = predict_stress_from_sensor(data)
                        print(f"✅ Prediction result: {result}")
                        last_values = current.copy()
                    else:
                        missing = [f for f in required_fields if f not in current]
                        print(f"❌ Missing required fields: {missing}")
                else:
                    print("📊 No change detected in sensor data.")
            else:
                print("⚠️  No sensor data found in Firebase.")
        except Exception as e:
            print(f"❌ Error while monitoring sensor data: {e}")
        time.sleep(5)

# --- FastAPI Startup Event ---
@app.on_event("startup")
def start_monitor():
    print("🚀 Starting Firebase monitoring for stress prediction...")
    threading.Thread(target=monitor_firebase_sensor_data, daemon=True).start()

# --- Health Check Endpoint ---
@app.get("/health")
def health_check():
    try:
        ref = db.reference("sensorData")
        current = ref.get()
        return {
            "status": "healthy",
            "firebase_connected": True,
            "current_sensor_data": current,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "firebase_connected": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# --- Manual Trigger Endpoint ---
@app.post("/trigger-prediction")
def trigger_prediction():
    try:
        ref = db.reference("sensorData")
        current = ref.get()
        if current:
            data = SensorData(
                mq135Voltage=float(current.get("mq135Voltage", 0.0)),
                humidity=float(current.get("humidity", 0.0)),
                soilMoisture=float(current.get("soilMoisture", 0.0)),
                temperature=float(current.get("temperature", 0.0))
            )
            result = predict_stress_from_sensor(data)
            return {"status": "success", "result": result, "input_data": current}
        else:
            return {"status": "error", "message": "No sensor data found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
