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
    flattened_eyeposition = [item for item in data.position_data]
    flattened_eyerotation = [item for item in data.rotation_data]

    # Create an instance of VRDataModel
    vr_data = VRDataModel(
        eyeposition=flattened_eyeposition,
        eyerotation=flattened_eyerotation,
        start=datetime.now(),
    )

    # Add the new record to the database
    db.add(vr_data)
    db.commit()
    db.refresh(vr_data)  # Refresh to get the generated ID

    return vr_data


@app.get("/getdata")
async def get_data(db: Session = Depends(get_db)):
    results = db.query(VRDataModel).all()
    for record in results:
        print(f"ID: {record.id}")
        print(f"Eye Position: {record.eyeposition}")
        print(f"Eye Rotation: {record.eyerotation}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=80)
