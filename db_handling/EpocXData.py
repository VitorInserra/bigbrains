from datetime import datetime
import pandas as pd
from sqlalchemy.orm import Session
from models.EpocXDataModel import EpocXDataModel



def insert_eeg_db(db: Session, session_id: str, df: pd.DataFrame):

    try:
        session_dump = {
            "start_stamp": datetime.fromtimestamp(df.iloc[0]["timestamp"]),
            "end_stamp": datetime.fromtimestamp(df.iloc[-1]["timestamp"]),
        }

        for key in df.columns:
            if key != "timestamp":
                session_dump[key] = list(df[key])

        eeg_data = EpocXDataModel(
            start_stamp=session_dump["start_stamp"],
            end_stamp=session_dump["end_stamp"],
            session_id=session_id,
            af3=session_dump.get("AF3", []),
            f7=session_dump.get("F7", []),
            f3=session_dump.get("F3", []),
            fc5=session_dump.get("FC5", []),
            t7=session_dump.get("T7", []),
            p7=session_dump.get("P7", []),
            o1=session_dump.get("O1", []),
            o2=session_dump.get("O2", []),
            p8=session_dump.get("P8", []),
            t8=session_dump.get("T8", []),
            fc6=session_dump.get("FC6", []),
            f4=session_dump.get("F4", []),
            f8=session_dump.get("F8", []),
            af4=session_dump.get("AF4", []),
        )

        db.add(eeg_data)
        db.commit()
        db.refresh(eeg_data)
        print(f"Saved EPOC X EEG data (session_id={session_id}) to database.")

    except Exception as e:
        print(f"No data or error encountered: {str(e)}")
