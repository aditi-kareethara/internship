from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import joblib

# 1. Initialize app
app = FastAPI(title="Palmer Archipelago Penguin Predictor API")

# 2. Enable CORS Configuration (Allows your Live Server HTML to talk to this port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Dynamic Absolute File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'random_forest_penguin_model.pkl')

try:
    model = joblib.load(model_path)
    print("=================== MODEL LOADED SUCCESSFULLY! ===================")
except Exception as e:
    print(f"=================== ERROR LOADING MODEL: {e} ===================")
    model = None

# 4. Pydantic Input Schema Structure
class PenguinFeatures(BaseModel):
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: int
    island_Biscoe: bool
    island_Dream: bool
    island_Torgersen: bool

# 5. Root route
@app.get("/")
def read_root():
    return {"status": "API is online and running smoothly!"}

# 6. Predict Route
@app.post("/predict")
def predict_penguin_species(data: PenguinFeatures):
    if model is None:
        return {"error": "Prediction model file is missing on the server."}
    
    try:
        # Explicit feature mapping in the exact training sequence order
        processed_data = {
            "culmen_length_mm": data.culmen_length_mm,
            "culmen_depth_mm": data.culmen_depth_mm,
            "flipper_length_mm": data.flipper_length_mm,
            "body_mass_g": data.body_mass_g,
            "sex": int(data.sex),
            "island_Biscoe": int(data.island_Biscoe),
            "island_Dream": int(data.island_Dream),
            "island_Torgersen": int(data.island_Torgersen)
        }
        
        input_df = pd.DataFrame([processed_data])
        numerical_prediction = int(model.predict(input_df)[0])
        
        species_mapping = {
            0: "Adelie",
            1: "Chinstrap",
            2: "Gentoo"
        }
        
        species_name = species_mapping.get(numerical_prediction, "Unknown Species")
        
        return {
            "prediction_class": numerical_prediction,
            "predicted_species": species_name
        }
        
    except Exception as e:
        return {"error": f"Prediction failed due to exception: {str(e)}"}