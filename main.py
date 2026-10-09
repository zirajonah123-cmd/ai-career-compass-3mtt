import os
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
from jose import JWTError, jwt

# -------------------------------------------------------------------
# CONFIGURATION & DATABASE SETUP
# -------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./attendance.db")
SECRET_KEY = os.getenv("SECRET_KEY", "SUPER_SECRET_KEY_CHANGE_IN_PRODUCTION")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

# Admin Default Credentials
ADMIN_USERNAME = "admin"
# Hashes to '1234' by default
ADMIN_PASSWORD_HASH = CryptContext(schemes=["bcrypt"], deprecated="auto").hash("1234")

# SQLAlchemy setup
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="admin/login")

# -------------------------------------------------------------------
# SQLALCHEMY MODELS
# -------------------------------------------------------------------
class DBStudent(Base):
    __tablename__ = "students"

    matric_no = Column(String, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    biometric_mode = Column(String, nullable=False)
    biometric_key = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    logs = relationship("DBAttendanceLog", back_populates="student")


class DBAttendanceLog(Base):
    __tablename__ = "attendance_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    matric_no = Column(String, ForeignKey("students.matric_no"), nullable=False)
    course = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="VERIFIED PRESENT")

    student = relationship("DBStudent", back_populates="logs")


Base.metadata.create_all(bind=engine)

# -------------------------------------------------------------------
# PYDANTIC SCHEMAS
# -------------------------------------------------------------------
class StudentRegisterRequest(BaseModel):
    matric_no: str
    full_name: str
    biometric_mode: str  # "camera" or "fingerprint"
    biometric_key: Optional[str] = None

class StudentResponse(BaseModel):
    matric_no: str
    full_name: str
    biometric_mode: str
    created_at: datetime

    class Config:
        from_attributes = True

class AttendanceSignRequest(BaseModel):
    matric_no: str
    course: str
    biometric_key: Optional[str] = None

class AttendanceLogResponse(BaseModel):
    id: int
    matric_no: str
    course: str
    timestamp: datetime
    status: str

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

# -------------------------------------------------------------------
# DEPENDENCIES & HELPERS
# -------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_admin(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate admin credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username != ADMIN_USERNAME:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

# -------------------------------------------------------------------
# FASTAPI APPLICATION & ENDPOINTS
# -------------------------------------------------------------------
app = FastAPI(
    title="Biometric Attendance API",
    description="Backend API supporting student enrollment, biometric authentication, and attendance tracking.",
    version="1.0.0",
)

# Enable CORS for cross-origin frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Biometric Attendance API Running"}

# 1. ADMIN AUTHENTICATION
@app.post("/admin/login", response_model=TokenResponse)
def admin_login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != ADMIN_USERNAME or not pwd_context.verify(form_data.password, ADMIN_PASSWORD_HASH):
        raise HTTPException(status_code=400, detail="Incorrect admin username or password")
    
    access_token = create_access_token(data={"sub": ADMIN_USERNAME})
    return {"access_token": access_token, "token_type": "bearer"}

# 2. REGISTER STUDENT
@app.post("/students/register", response_model=StudentResponse, status_code=status.HTTP_201_CREATED)
def register_student(payload: StudentRegisterRequest, db: Session = Depends(get_db)):
    matric_clean = payload.matric_no.strip().upper()
    
    existing = db.query(DBStudent).filter(DBStudent.matric_no == matric_clean).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Student with Matric {matric_clean} already registered.")
    
    student = DBStudent(
        matric_no=matric_clean,
        full_name=payload.full_name.strip(),
        biometric_mode=payload.biometric_mode,
        biometric_key=payload.biometric_key
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return student

# 3. SIGN ATTENDANCE
@app.post("/attendance/sign", response_model=AttendanceLogResponse)
def sign_attendance(payload: AttendanceSignRequest, db: Session = Depends(get_db)):
    matric_clean = payload.matric_no.strip().upper()
    student = db.query(DBStudent).filter(DBStudent.matric_no == matric_clean).first()

    if not student:
        raise HTTPException(status_code=404, detail="Student not found. Please register first.")

    # Log record directly
    log = DBAttendanceLog(
        matric_no=matric_clean,
        course=payload.course.strip(),
        status="VERIFIED PRESENT"
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log

# 4. ADMIN: GET ALL REGISTERED STUDENTS
@app.get("/admin/students", response_model=List[StudentResponse])
def get_all_students(admin: str = Depends(verify_admin), db: Session = Depends(get_db)):
    return db.query(DBStudent).order_by(DBStudent.created_at.desc()).all()

# 5. ADMIN: GET ALL ATTENDANCE LOGS
@app.get("/admin/logs", response_model=List[AttendanceLogResponse])
def get_all_logs(admin: str = Depends(verify_admin), db: Session = Depends(get_db)):
    return db.query(DBAttendanceLog).order_by(DBAttendanceLog.timestamp.desc()).all()
