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
from muselsl import stream, list_muses
import pandas as pd
import time
from collections import deque
from contextlib import asynccontextmanager

BUFFER_SIZE = 256*10
eeg_buffer = deque(maxlen=BUFFER_SIZE)
buffer_lock = threading.Lock()

def continuous_stream():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def on_data_received(timestamp, data, aux, markers):
        with buffer_lock:
            eeg_buffer.append((timestamp, data))

    while True:
        try:
            muses = list_muses()
            if muses:
                stream(address=muses[0]['address'], callback=on_data_received)
            else:
                time.sleep(5)
        except Exception as e:
            time.sleep(5)

def extract_recent_eeg_data():
    with buffer_lock:
        return list(eeg_buffer)

def save_eeg_data_to_file_and_db(db: Session):
    data = extract_recent_eeg_data()
    if not data:
        print("No data available to save.")
        return

    print(f"Saving {len(data)} rows of EEG data.")  # Log number of rows
    df = pd.DataFrame(data, columns=["timestamp", "data"])
    channel_data = pd.DataFrame(df["data"].tolist(), columns=["tp9", "af7", "af8", "tp10", "aux"])
    df = pd.concat([df["timestamp"], channel_data], axis=1)

    filename = f"eeg_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved EEG data to {filename}")

    insert_eeg_db(db, df)

class DataDumpRequest(BaseModel):
    start_stamp: datetime
    eye_id: str
    position_data: List[List[float]]
    rotation_data: List[List[float]]
    end_stamp: datetime

@asynccontextmanager
async def lifespan(app: FastAPI):
    stream_thread = threading.Thread(target=continuous_stream, daemon=True)
    stream_thread.start()
    yield

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

    background_tasks.add_task(save_eeg_data_to_file_and_db, db)
    return {"message": "VR data saved and recent EEG data recorded."}

@app.get("/stream")
async def eeg_stream(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    background_tasks.add_task(save_eeg_data_to_file_and_db, db)
    return {"message": "Recent EEG data saved."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8083, reload=False)
