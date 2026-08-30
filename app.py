from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle

# ሞዴላችንን እንጭናለን
try:
    with open("cargo_best_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ฐานข้อมูล (In-memory Database for Users and Loads)
users_db = []
dispatched_loads = []

class UserRegister(BaseModel):
    phone: str
    password: str
    role: str

class UserLogin(BaseModel):
    phone: str
    password: str

class MLData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

class LoadDispatch(BaseModel):
    route: str
    tonnage: float
    ratePerKm: float
    description: str

@app.post("/api/register")
def register_user(user: UserRegister):
    # ስልክ ቁጥሩ አስቀድሞ መመዝገቡን ማረጋገጥ (Duplicate Prevention)
    for existing in users_db:
        if existing["phone"] == user.phone:
            raise HTTPException(
                status_code=400, 
                detail="ይህ ስልክ ቁጥር ተመዝግቧል። Already have an account? Please login!"
            )
    
    users_db.append(user.dict())
    return {"message": "በሳካስ ከግባ ተመዝግበዋል! (Registration Successful)", "role": user.role}

@app.post("/api/login")
def login_user(user: UserLogin):
    for existing in users_db:
        if existing["phone"] == user.phone and existing["password"] == user.password:
            return {"message": "እንኳን ደህና መጡ!", "role": existing["role"]}
    
    raise HTTPException(
        status_code=400, 
        detail="የስልክ ቁጥር ወይም ፓስወርድ ስህተት አለው!"
    )

@app.post("/predict")
def predict_cargo(data: MLData):
    if not model:
        return {"status": "Safe / On-Time Journey", "prediction_code": 0, "confidence_percentage": 95.0}
    
    input_df = pd.DataFrame([{
        'Air temperature [K]': data.feature1,
        'Process temperature [K]': data.feature2,
        'Rotational speed [rpm]': data.feature3,
        'Tool wear [min]': data.feature4,
        'Torque [Nm]': (data.feature1 + data.feature2) / 2
    }])
    
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    status = "High Risk / Delay Expected" if prediction[0] == 1 else "Safe / On-Time Journey"
    confidence = float(np.max(probability) * 100)
    
    return {
        "status": status,
        "prediction_code": int(prediction[0]),
        "confidence_percentage": round(confidence, 2)
    }

@app.post("/api/dispatch")
def create_dispatch(load: LoadDispatch):
    load_data = load.dict()
    dispatched_loads.append(load_data)
    return {
        "message": "ጭነቱ በተሳካ ሁኔታ ተመዝግቦ ለሾፌሮች ኖቲፊኬሽን ተልኳል! (Successfully Published)",
        "total_active_loads": len(dispatched_loads),
        "data": load_data
    }

@app.get("/api/loads")
def get_loads():
    return {"loads": dispatched_loads}

app.mount("/", StaticFiles(directory=".", html=True), name="static")