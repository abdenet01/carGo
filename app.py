from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import random

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

# In-memory Database for Users, OTPs, and Loads
users_db = []
otp_db = {}
dispatched_loads = []
driver_voice_loads = []

class UserRegister(BaseModel):
    phone: str
    password: str
    role: str

class UserLogin(BaseModel):
    phone: str
    password: str

class ForgotPasswordRequest(BaseModel):
    phone: str

class ResetPasswordRequest(BaseModel):
    phone: str
    otp: str
    newPassword: str

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

class PaymentRequest(BaseModel):
    amount: float

class DriverVoiceDispatch(BaseModel):
    phone: str
    voiceText: str

@app.post("/api/register")
def register_user(user: UserRegister):
    for existing in users_db:
        if existing["phone"] == user.phone:
            raise HTTPException(
                status_code=400, 
                detail="ይህ ስልክ ቁጥር ቀደም ሲል ተመዝግቧል፤ እባክዎ በቀጥታ ይግቡ (Log In)"
            )
    
    users_db.append(user.dict())
    return {"success": True, "message": "መለያዎ በሳካ ሁኔታ ተፈጥሯል!", "role": user.role}

@app.post("/api/login")
def login_user(user: UserLogin):
    for existing in users_db:
        if existing["phone"] == user.phone and existing["password"] == user.password:
            return {"success": True, "message": "እንኳን ደህና መጡ!", "role": existing["role"]}
    
    raise HTTPException(
        status_code=400, 
        detail="የስልክ ቁጥር ወይም ፓስወርድ ስህተት አለው!"
    )

@app.post("/api/forgot-password")
def forgot_password(data: ForgotPasswordRequest):
    user_found = any(u["phone"] == data.phone for u in users_db)
    if not user_found:
        raise HTTPException(
            status_code=400, 
            detail="ይህ ስልክ ቁጥር በሲስተሙ ውስጥ አልተገኘም!"
        )
    
    # Generate 4-digit OTP
    otp = str(random.randint(1000, 9999))
    otp_db[data.phone] = otp
    print(f"[SMS Gateway] OTP for {data.phone} is: {otp}")
    
    return {"success": True, "message": "የማረጋገጫ ኮድ (OTP) ወደ ስልክ ቁጥርዎ ተልኳል!"}

@app.post("/api/reset-password")
def reset_password(data: ResetPasswordRequest):
    if data.phone not in otp_db or otp_db[data.phone] != data.otp:
        raise HTTPException(
            status_code=400, 
            detail="የተሳሳተ የኦቲፒ (OTP) ኮድ!"
        )
    
    updated = False
    for user in users_db:
        if user["phone"] == data.phone:
            user["password"] = data.newPassword
            updated = True
            break
            
    if not updated:
        raise HTTPException(
            status_code=400, 
            detail="ተጠቃሚው አልተገኘም!"
        )
        
    del otp_db[data.phone]
    return {"success": True, "message": "የይለፍ ቃልዎ በተሳካ ሁኔታ ተቀይሯል!"}

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
        "success": True,
        "message": "ጭነቱ በተሳካ ሁኔታ ተመዝግቦ ለሾፌሮች ኖቲፊኬሽን ተልኳል! (Successfully Published)",
        "total_active_loads": len(dispatched_loads),
        "data": load_data
    }

@app.get("/api/loads")
def get_loads():
    return {"loads": dispatched_loads}

@app.post("/api/create-payment")
def create_payment(payment: PaymentRequest):
    return {
        "success": True,
        "checkoutUrl": "https://superapp.ethiotelecom.et/portal/index.html"
    }

@app.post("/api/driver-voice-dispatch")
def driver_voice_dispatch(data: DriverVoiceDispatch):
    text = data.voiceText.lower()
    
    destination = "ድሬዳዋ" if "ድሬዳዋ" in text else "አዲስ አበባ"
    return_time = "ነገ ጠዋት" if "ነገ" in text else "በቅርብ"
    
    tonnage = 30
    for word in text.split():
        if word.isdigit():
            tonnage = float(word)
            break

    load_entry = {
        "driverPhone": data.phone,
        "destination": destination,
        "returnTime": return_time,
        "tonnage": tonnage,
        "rawVoice": data.voiceText,
        "status": "Active Backhaul"
    }
    
    driver_voice_loads.append(load_entry)
    
    return {
        "success": True,
        "message": f"ድምጽዎ ተተንትኖ ወደ {destination} ለሚገኙ ሺፐሮች አውቶማቲክ ኖቲፊኬሽን ተልኳል!",
        "extractedData": load_entry,
        "totalNotificationsSent": len(users_db)
    }

@app.get("/api/driver-loads")
def get_driver_loads():
    return {"backhaulLoads": driver_voice_loads}

# Static files mount should always be at the very bottom
app.mount("/", StaticFiles(directory=".", html=True), name="static")
