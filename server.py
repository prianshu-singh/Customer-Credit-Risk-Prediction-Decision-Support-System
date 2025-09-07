from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta, timezone
import jwt
from passlib.context import CryptContext
import pandas as pd
import io
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import asyncio
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# JWT Configuration
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'fallback-secret-key')
JWT_ALGORITHM = os.environ.get('JWT_ALGORITHM', 'HS256')
JWT_EXPIRATION_HOURS = int(os.environ.get('JWT_EXPIRATION_HOURS', 24))

# Create the main app
app = FastAPI(title="Credit Risk Prediction System")
api_router = APIRouter(prefix="/api")

# Pydantic Models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Token(BaseModel):
    access_token: str
    token_type: str

class CustomerData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str
    name: str
    age: int
    annual_income: float
    credit_score: int
    debt_to_income_ratio: float
    employment_length: int
    loan_amount: float
    loan_purpose: str
    home_ownership: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class RiskPrediction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str
    risk_score: float
    risk_level: str
    ai_analysis: str
    ml_probability: float
    recommendations: List[str]
    alert_generated: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Alert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str
    risk_score: float
    message: str
    severity: str
    status: str = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Utility Functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        
        user = await db.users.find_one({"username": username})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return User(**user)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

def prepare_for_mongo(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
    return data

def parse_from_mongo(item):
    if isinstance(item, dict):
        for key, value in item.items():
            if isinstance(value, str) and 'T' in value:
                try:
                    item[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except:
                    pass
    return item

# ML Model Functions
def train_ml_model(data_df):
    """Train a simple ML model for risk prediction"""
    # Prepare features
    feature_cols = ['age', 'annual_income', 'credit_score', 'debt_to_income_ratio', 'employment_length', 'loan_amount']
    
    # Create a synthetic risk label based on common risk factors
    data_df['risk_label'] = (
        (data_df['credit_score'] < 600) | 
        (data_df['debt_to_income_ratio'] > 0.4) |
        (data_df['annual_income'] < 30000)
    ).astype(int)
    
    X = data_df[feature_cols].fillna(0)
    y = data_df['risk_label']
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

async def analyze_with_ai(customer_data, ml_probability):
    """Use OpenAI GPT-5 for risk analysis"""
    try:
        chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=f"risk_analysis_{customer_data['customer_id']}",
            system_message="You are a financial risk analyst. Analyze customer data and provide risk assessment with specific recommendations."
        ).with_model("openai", "gpt-5")
        
        customer_profile = f"""
        Customer Profile:
        - Name: {customer_data['name']}
        - Age: {customer_data['age']}
        - Annual Income: ${customer_data['annual_income']:,.2f}
        - Credit Score: {customer_data['credit_score']}
        - Debt-to-Income Ratio: {customer_data['debt_to_income_ratio']:.2%}
        - Employment Length: {customer_data['employment_length']} years
        - Loan Amount: ${customer_data['loan_amount']:,.2f}
        - Loan Purpose: {customer_data['loan_purpose']}
        - Home Ownership: {customer_data['home_ownership']}
        - ML Model Risk Probability: {ml_probability:.2%}
        """
        
        user_message = UserMessage(
            text=f"""Analyze this customer's credit risk profile and provide:
            1. Overall risk assessment (Low/Medium/High)
            2. Key risk factors identified
            3. 3-5 specific recommendations for risk mitigation
            4. Risk score out of 100 (higher = more risk)
            
            {customer_profile}
            
            Please be concise and actionable in your analysis."""
        )
        
        response = await chat.send_message(user_message)
        return response
        
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

# Authentication Routes
@api_router.post("/register", response_model=User)
async def register_user(user_data: UserCreate):
    # Check if user exists
    existing_user = await db.users.find_one({"username": user_data.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    existing_email = await db.users.find_one({"email": user_data.email})
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    hashed_password = get_password_hash(user_data.password)
    user = User(username=user_data.username, email=user_data.email)
    user_dict = user.dict()
    user_dict['password'] = hashed_password
    user_dict = prepare_for_mongo(user_dict)
    
    await db.users.insert_one(user_dict)
    return user

@api_router.post("/login", response_model=Token)
async def login_user(login_data: UserLogin):
    user = await db.users.find_one({"username": login_data.username})
    if not user or not verify_password(login_data.password, user['password']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user['username']})
    return {"access_token": access_token, "token_type": "bearer"}

# File Upload and Processing Routes
@api_router.post("/upload-customer-data")
async def upload_customer_data(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="File must be CSV or Excel format")
    
    try:
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        else:
            df = pd.read_excel(io.BytesIO(contents))
        
        # Validate required columns
        required_cols = ['customer_id', 'name', 'age', 'annual_income', 'credit_score', 
                        'debt_to_income_ratio', 'employment_length', 'loan_amount', 
                        'loan_purpose', 'home_ownership']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_cols)}"
            )
        
        # Process and store customer data
        customers_processed = 0
        for _, row in df.iterrows():
            customer_data = CustomerData(**row.to_dict())
            customer_dict = prepare_for_mongo(customer_data.dict())
            
            # Check if customer already exists
            existing = await db.customers.find_one({"customer_id": customer_data.customer_id})
            if not existing:
                await db.customers.insert_one(customer_dict)
                customers_processed += 1
        
        return {
            "message": f"Successfully processed {customers_processed} customers",
            "total_rows": len(df),
            "processed": customers_processed
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

# Risk Analysis Routes
@api_router.post("/analyze-risk/{customer_id}")
async def analyze_customer_risk(
    customer_id: str,
    current_user: User = Depends(get_current_user)
):
    # Get customer data
    customer = await db.customers.find_one({"customer_id": customer_id})
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    try:
        # Get all customers for ML training
        all_customers = await db.customers.find().to_list(1000)
        if len(all_customers) < 5:
            raise HTTPException(status_code=400, detail="Need at least 5 customers for ML analysis")
        
        df = pd.DataFrame(all_customers)
        
        # Train ML model
        model, scaler = train_ml_model(df)
        
        # Predict for current customer
        feature_cols = ['age', 'annual_income', 'credit_score', 'debt_to_income_ratio', 'employment_length', 'loan_amount']
        customer_features = [[customer[col] for col in feature_cols]]
        customer_features_scaled = scaler.transform(customer_features)
        
        ml_probability = model.predict_proba(customer_features_scaled)[0][1]  # Probability of high risk
        
        # Get AI analysis
        ai_analysis = await analyze_with_ai(customer, ml_probability)
        
        # Determine risk level and score
        risk_score = min(100, max(0, ml_probability * 100 + np.random.normal(0, 5)))  # Add some variation
        
        if risk_score >= 70:
            risk_level = "High"
        elif risk_score >= 40:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Generate recommendations
        recommendations = []
        if customer['credit_score'] < 650:
            recommendations.append("Improve credit score through timely payments")
        if customer['debt_to_income_ratio'] > 0.3:
            recommendations.append("Reduce debt-to-income ratio")
        if customer['annual_income'] < 40000:
            recommendations.append("Consider income verification and stability")
        if customer['employment_length'] < 2:
            recommendations.append("Monitor employment stability")
        
        if not recommendations:
            recommendations.append("Maintain current financial practices")
        
        # Create risk prediction
        risk_prediction = RiskPrediction(
            customer_id=customer_id,
            risk_score=risk_score,
            risk_level=risk_level,
            ai_analysis=ai_analysis,
            ml_probability=ml_probability,
            recommendations=recommendations,
            alert_generated=risk_level == "High"
        )
        
        # Save prediction
        prediction_dict = prepare_for_mongo(risk_prediction.dict())
        await db.risk_predictions.insert_one(prediction_dict)
        
        # Generate alert if high risk
        if risk_level == "High":
            alert = Alert(
                customer_id=customer_id,
                risk_score=risk_score,
                message=f"High risk customer detected: {customer['name']} (Score: {risk_score:.1f})",
                severity="high"
            )
            alert_dict = prepare_for_mongo(alert.dict())
            await db.alerts.insert_one(alert_dict)
        
        return risk_prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing risk: {str(e)}")

@api_router.get("/customers", response_model=List[CustomerData])
async def get_customers(current_user: User = Depends(get_current_user)):
    customers = await db.customers.find().to_list(1000)
    return [CustomerData(**parse_from_mongo(customer)) for customer in customers]

@api_router.get("/risk-predictions", response_model=List[RiskPrediction])
async def get_risk_predictions(current_user: User = Depends(get_current_user)):
    predictions = await db.risk_predictions.find().to_list(1000)
    return [RiskPrediction(**parse_from_mongo(prediction)) for prediction in predictions]

@api_router.get("/alerts", response_model=List[Alert])
async def get_alerts(current_user: User = Depends(get_current_user)):
    alerts = await db.alerts.find({"status": "active"}).to_list(1000)
    return [Alert(**parse_from_mongo(alert)) for alert in alerts]

@api_router.put("/alerts/{alert_id}/dismiss")
async def dismiss_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user)
):
    result = await db.alerts.update_one(
        {"id": alert_id},
        {"$set": {"status": "dismissed"}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return {"message": "Alert dismissed"}

@api_router.get("/dashboard-stats")
async def get_dashboard_stats(current_user: User = Depends(get_current_user)):
    total_customers = await db.customers.count_documents({})
    total_predictions = await db.risk_predictions.count_documents({})
    active_alerts = await db.alerts.count_documents({"status": "active"})
    
    # Risk distribution
    high_risk = await db.risk_predictions.count_documents({"risk_level": "High"})
    medium_risk = await db.risk_predictions.count_documents({"risk_level": "Medium"})
    low_risk = await db.risk_predictions.count_documents({"risk_level": "Low"})
    
    return {
        "total_customers": total_customers,
        "total_predictions": total_predictions,
        "active_alerts": active_alerts,
        "risk_distribution": {
            "high": high_risk,
            "medium": medium_risk,
            "low": low_risk
        }
    }

# Health check
@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc)}

# Include router
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()