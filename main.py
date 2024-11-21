# main.py
import threading
import asyncio
from fastapi import FastAPI, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from db import get_db
from models.models import VRDataModel, EEGDataModel
from entities.EEGData import insert_eeg_db
from pydantic import BaseModel
from typing import List
from datetime import datetime
import uvicorn
from muselsl import stream, list_muses, record
import pandas as pd
import os
import time
from contextlib import asynccontextmanager


def stream_muse():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    while True:
        try:
            muses = list_muses()
            if muses:
                try:
                    print("Starting Muse stream...")
                    stream(muses[0]["address"])
                except Exception as e:
                    print(f"Error streaming Muse: {e}")
                    time.sleep(5)
            else:
                print("No Muse devices found. Retrying in 5 seconds...")
                time.sleep(5)
        except Exception as e:
            print(f"Error in stream_muse: {e}")
            time.sleep(5)

def call_record_and_insert_eeg_db(db: Session):
    directory = os.getcwd()
    filename = os.path.join(directory, f"session_data.csv")
    
    print("Starting EEG recording for 10 seconds")
    record(duration=10, filename=filename)
    print("Finished EEG recording")

    insert_eeg_db(db)

    print("EEG data inserted into the database and temporary file removed.")

class DataDumpRequest(BaseModel):
    start_stamp: datetime
    eye_id: str
    position_data: List[List[float]]
    rotation_data: List[List[float]]
    end_stamp: datetime

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    muse_thread = threading.Thread(target=stream_muse, daemon=True)
    muse_thread.start()
    print("Started streaming Muse data.")
    yield
    print("Application shutdown")

app = FastAPI(lifespan=lifespan)

@app.post("/datadump")
async def data_dump(data: DataDumpRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    start_stamp = data.start_stamp
    eye_id = data.eye_id
    flattened_eyeposition = data.position_data
    flattened_eyerotation = data.rotation_data
    end_stamp = data.end_stamp

    vr_data = VRDataModel(
        start_stamp=start_stamp,
        eye_id=eye_id,
        eyeposition=flattened_eyeposition,
        eyerotation=flattened_eyerotation,
        end_stamp=end_stamp
    )

    db.add(vr_data)
    db.commit()
    db.refresh(vr_data)

    print("Saved VR data to db.")
    
    background_tasks.add_task(call_record_and_insert_eeg_db, db)
    
    return {"message": "VR data saved and EEG recording started."}

@app.get("/getdata")
async def get_data(db: Session = Depends(get_db)):
    results = db.query(VRDataModel).all()
    data_list = []
    for record in results:
        data_list.append({
            "ID": record.id,
            "Eye Position": record.eyeposition,
            "Eye Rotation": record.eyerotation
        })
    return data_list

@app.get("/stream")
async def eeg_stream(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    background_tasks.add_task(call_record_and_insert_eeg_db, db)
    print("Started EEG recording")
    return {"message": "EEG recording started"}

if __name__ == "__main__":
    f = open("session_data.csv", "w")
    f.truncate()
    f.write("timestamp,tp9,af7,af8,tp10,hr")
    f.close()
    uvicorn.run("main:app", host="0.0.0.0", port=8083, reload=False)
