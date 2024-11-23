import os
import asyncio
from fastapi import FastAPI, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from db import get_db
from models.models import VRDataModel, MuseDataModel
from entities.EEGData import insert_eeg_db
from pydantic import BaseModel
from typing import List
from datetime import datetime
import uvicorn
from muselsl import stream, list_muses, record
import pandas as pd
import time
from contextlib import asynccontextmanager
import multiprocessing as mp
import threading


app = FastAPI()

def stream_muse():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        try:
            muses = list_muses()
            if muses:
                print(f"Connecting to Muse: {muses[0]['name']}...")
                stream(muses[0]['address'])
            else:
                print("No Muse devices found. Retrying in 5 seconds...")
                time.sleep(5)
        except Exception as e:
            print(f"Error in Muse streaming: {e}. Retrying in 5 seconds...")
            time.sleep(5)

def start_muse_streaming():
    muse_thread = mp.Process(target=stream_muse, daemon=True)
    muse_thread.start()
    print("Muse streaming process started.")
    return muse_thread

class DataDumpRequest(BaseModel):
    start_stamp: datetime
    eye_id: str
    position_data: List[List[float]]
    rotation_data: List[List[float]]
    end_stamp: datetime

@app.post("/datadump")
async def data_dump(data: DataDumpRequest, db: Session = Depends(get_db)):
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

    # background_tasks.add_task(save_eeg_data_to_file_and_db, db)
    return {"message": "VR data saved and recent EEG data recorded."}


async def eeg_stream(db: Session = Depends(get_db)):
    p = mp.Process(target=call_record)
    p.start()
    print("Started recording Muse")

@app.get("/record")
def call_record():
    directory = os.getcwd()
    filename = os.path.join(directory, "session_data.csv")
    record(duration=0, filename=filename)
    print("Finished recording Muse")


@app.get("/db-insert-eeg")
async def db_insert_eeg(db: Session = Depends(get_db)):
    insert_eeg_db(db)


def init_record():
    while True:
        f = open("session_data.csv", "w")
        f.truncate()
        f.write("timestamp,tp9,af7,af8,tp10,hr\n")
        f.close()
        start_muse_streaming()
        time.sleep(15)
        input("Click enter to start recording:")
        record_proc = mp.Process(target=call_record, daemon=True)
        record_proc.start()
        input("Click enter to stop recording:")
        record_proc.kill()

if __name__ == "__main__":
    # t = threading.Thread(target=init_record)
    # t.start()
    uvicorn.run("main:app", host="0.0.0.0", port=8083, reload=False)
