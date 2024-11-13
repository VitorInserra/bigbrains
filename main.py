# main.py
import multiprocessing.process
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from db import get_db
from models.models import VRDataModel, EEGDataModel
from entities.EEGData import insert_eeg_db
from pydantic import BaseModel
from typing import List
from datetime import datetime
import uvicorn
from muselsl import stream, list_muses, record
import multiprocessing
import pandas as pd
import os


app = FastAPI()



# Define the Pydantic model for the incoming request
class DataDumpRequest(BaseModel):
    start_stamp: datetime
    position_data: List[List[float]]
    rotation_data: List[List[float]]
    end_stamp: datetime 


@app.post("/datadump")
async def data_dump(data: DataDumpRequest, db: Session = Depends(get_db)):
    # Flatten the position_data and rotation_data listsed

    start_stamp = data.start_stamp
    flattened_eyeposition = [item for item in data.position_data]
    flattened_eyerotation = [item for item in data.rotation_data]
    end_stamp = data.end_stamp

    # Create an instance of VRDataModel
    vr_data = VRDataModel(
        start_stamp=start_stamp,
        eyeposition=flattened_eyeposition,
        eyerotation=flattened_eyerotation,
        end_stamp=end_stamp
    )

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

 
def stream_muse():
    muses = list_muses()
    try:
        stream(muses[0]["address"])
    except:
        print("Skipped streaming.")


@app.get("/stream")
async def stream(db: Session = Depends(get_db)):
    
    directory = os.getcwd()
    filename = os.path.join(directory, "session_data.csv")
    p = multiprocessing.Process(target=record, args=(10, filename, ))
    p.start()
    p.join()
    insert_eeg_db(db)

@app.get("/db-insert-eeg")
async def db_insert_eeg(db: Session = Depends(get_db)):
    insert_eeg_db(db)


if __name__ == "__main__":
    p = multiprocessing.Process(target=stream_muse)
    p.start()
    uvicorn.run("main:app", host="0.0.0.0", port=8083)
    # uvicorn.run("main:app", port=8083)
