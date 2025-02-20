import os
import asyncio
from fastapi import FastAPI, Depends, Request
from sqlalchemy.orm import Session
from db import get_db
from models.VRDataModel import VRDataModel
from db_handling.EpocXData import insert_eeg_db
import uvicorn
import pandas as pd
import time
import multiprocessing as mp
import threading
import uuid
from contextlib import asynccontextmanager
import muse_record
import EpocX
from db_handling.VRData import VRData
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List
from datetime import datetime

global_session_id: str = None


def set_global_session_id():
    """Initialize the global session ID."""
    global global_session_id
    global_session_id = str(uuid.uuid4())
    print(f"Initialized global session_id: {global_session_id}")


def get_global_session_id():
    """Retrieve the global session ID."""
    global global_session_id
    if not global_session_id:
        raise ValueError("Global session_id is not set.")
    return global_session_id


async def eeg_stream(db: Session = Depends(get_db)):
    p = mp.Process(target=call_record)
    p.start()
    print("Started recording Muse")


app = FastAPI()


@app.get("/")
async def set_session_id():
    start = time.time()
    set_global_session_id()
    t = threading.Thread(target=init_epoc_record)
    t.start()
    time.sleep(3)
    print(time.time() - start)
    return


def init_epoc_record():
    asyncio.run(EpocX.main())


def init_muse_record():
    while True:
        f = open("session_data.csv", "w")
        f.truncate()
        f.write("timestamp,tp9,af7,af8,tp10,hr\n")
        f.close()
        muse_record.start_muse_streaming()
        time.sleep(3)
        input("Click enter to start recording:")
        record_proc = mp.Process(target=call_record, daemon=True)
        record_proc.start()
        time.sleep(10)
        input("Click enter to stop recording:")
        record_proc.kill()
        print("Recording stopped.")



@app.get("/muse-record")
def call_muse_record():
    directory = os.getcwd()
    filename = os.path.join(directory, "session_data.csv")
    record(duration=0, filename=filename)
    print("Finished recording Muse")


@app.get("/db-insert-eeg")
async def db_insert_eeg(db: Session = Depends(get_db)):
    session_id = get_global_session_id()
    df = pd.DataFrame(EpocX.pow_data_batch)
    print(df)
    EpocX.pow_data_batch.clear()
    insert_eeg_db(db, session_id, df)


@app.post("/datadump")
async def data_dump(data: VRData, db: Session = Depends(get_db)):
    start_stamp = data.start_stamp
    eye_id = data.eye_id
    flattened_eyeposition = data.eyeposition
    flattened_eyerotation = data.eyerotation
    end_stamp = data.end_stamp
    score = data.score
    test_version = data.test_version
    end_timer = data.end_timer
    initial_timer = data.initial_timer
    rotation_speed = data.rotation_speed
    obj_rotation = data.obj_rotation

    vr_data = VRDataModel(
        start_stamp=start_stamp,
        session_id=get_global_session_id(),
        eye_id=eye_id,
        eyeposition=flattened_eyeposition,
        eyerotation=flattened_eyerotation,
        end_stamp=end_stamp,
        score=score,
        test_version=test_version,
        end_timer=end_timer,
        initial_timer=initial_timer,
        rotation_speed=rotation_speed,
        obj_rotation=obj_rotation,
        description="",
    )

    db.add(vr_data)
    db.commit()
    db.refresh(vr_data)

    return {"message": "VR data saved and recent EEG data recorded."}


@app.get("/blink-sync")
async def set_session_id():
    start = time.time()
    set_global_session_id()
    t = threading.Thread(target=init_blink_sync)
    t.start()
    time.sleep(3)
    print(time.time() - start)
    return

def init_blink_sync():
    asyncio.run(EpocX.blink_sync())

@app.post("/compare-blinks/{left}/{ts_l}/{right}/{ts_r}")
async def compare_blinks(left: bool, ts_l: datetime, right: bool, ts_r: datetime):
    print(left, ts_l)
    print(right, ts_r)

    return {"message": "ok"}



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8083, reload=False)
