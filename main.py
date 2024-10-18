# main.py
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from db import SessionLocal
from models.models import VRDataModel  # Import your model from models.py
from pydantic import BaseModel
from typing import List
from datetime import datetime
import uvicorn

app = FastAPI()

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Define the Pydantic model for the incoming request
class DataDumpRequest(BaseModel):
    position_data: List[List[float]]
    rotation_data: List[List[float]]

@app.post("/datadump")
async def data_dump(data: DataDumpRequest, db: Session = Depends(get_db)):
    # Flatten the position_data and rotation_data lists
    print(data.position_data)
    flattened_eyeposition = [item for sublist in data.position_data for item in sublist]
    flattened_eyerotation = [item for sublist in data.rotation_data for item in sublist]

    # Create an instance of VRDataModel
    vr_data = VRDataModel(
        eyeposition=flattened_eyeposition,
        eyerotation=flattened_eyerotation,
        start=datetime.now()
    )

    # Add the new record to the database
    db.add(vr_data)
    db.commit()
    db.refresh(vr_data)  # Refresh to get the generated ID

    return vr_data

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=80)