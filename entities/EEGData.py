from datetime import datetime
import pandas as pd
from sqlalchemy.orm import Session
from models.models import EEGDataModel


def edit_names(text):
    text = str(text)
    text = text.replace("'", "''")

    return text


def insert_eeg_db(db: Session, df: pd.DataFrame):
    try:
        session_dump = {
            'start_stamp': datetime.fromtimestamp(df.iloc[0]['timestamp']),
            'end_stamp': datetime.fromtimestamp(df.iloc[-1]['timestamp']),
            'tp9': df['tp9'].tolist(),
            'af7': df['af7'].tolist(),
            'af8': df['af8'].tolist(),
            'tp10': df['tp10'].tolist(),
        }

        eeg_data = EEGDataModel(
            start_stamp=session_dump['start_stamp'],
            tp9=session_dump['tp9'],
            af7=session_dump['af7'],
            af8=session_dump['af8'],
            tp10=session_dump['tp10'],
            end_stamp=session_dump['end_stamp'],
        )

        db.add(eeg_data)
        db.commit()
        db.refresh(eeg_data)
    except Exception as e:
        pass