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
            session_id=session_id,
            start_stamp=session_dump["start_stamp"],
            end_stamp=session_dump["end_stamp"],
            # AF3
            af3_theta=session_dump.get("AF3_theta", []),
            af3_alpha=session_dump.get("AF3_alpha", []),
            af3_beta_l=session_dump.get("AF3_betaL", []),
            af3_beta_h=session_dump.get("AF3_betaH", []),
            af3_gamma=session_dump.get("AF3_gamma", []),
            # F7
            f7_theta=session_dump.get("F7_theta", []),
            f7_alpha=session_dump.get("F7_alpha", []),
            f7_beta_l=session_dump.get("F7_betaL", []),
            f7_beta_h=session_dump.get("F7_betaH", []),
            f7_gamma=session_dump.get("F7_gamma", []),
            # F3
            f3_theta=session_dump.get("F3_theta", []),
            f3_alpha=session_dump.get("F3_alpha", []),
            f3_beta_l=session_dump.get("F3_betaL", []),
            f3_beta_h=session_dump.get("F3_betaH", []),
            f3_gamma=session_dump.get("F3_gamma", []),
            # FC5
            fc5_theta=session_dump.get("FC5_theta", []),
            fc5_alpha=session_dump.get("FC5_alpha", []),
            fc5_beta_l=session_dump.get("FC5_betaL", []),
            fc5_beta_h=session_dump.get("FC5_betaH", []),
            fc5_gamma=session_dump.get("FC5_gamma", []),
            # T7
            t7_theta=session_dump.get("T7_theta", []),
            t7_alpha=session_dump.get("T7_alpha", []),
            t7_beta_l=session_dump.get("T7_betaL", []),
            t7_beta_h=session_dump.get("T7_betaH", []),
            t7_gamma=session_dump.get("T7_gamma", []),
            # P7
            p7_theta=session_dump.get("P7_theta", []),
            p7_alpha=session_dump.get("P7_alpha", []),
            p7_beta_l=session_dump.get("P7_betaL", []),
            p7_beta_h=session_dump.get("P7_betaH", []),
            p7_gamma=session_dump.get("P7_gamma", []),
            # O1
            o1_theta=session_dump.get("O1_theta", []),
            o1_alpha=session_dump.get("O1_alpha", []),
            o1_beta_l=session_dump.get("O1_betaL", []),
            o1_beta_h=session_dump.get("O1_betaH", []),
            o1_gamma=session_dump.get("O1_gamma", []),
            # O2
            o2_theta=session_dump.get("O2_theta", []),
            o2_alpha=session_dump.get("O2_alpha", []),
            o2_beta_l=session_dump.get("O2_betaL", []),
            o2_beta_h=session_dump.get("O2_betaH", []),
            o2_gamma=session_dump.get("O2_gamma", []),
            # P8
            p8_theta=session_dump.get("P8_theta", []),
            p8_alpha=session_dump.get("P8_alpha", []),
            p8_beta_l=session_dump.get("P8_betaL", []),
            p8_beta_h=session_dump.get("P8_betaH", []),
            p8_gamma=session_dump.get("P8_gamma", []),
            # T8
            t8_theta=session_dump.get("T8_theta", []),
            t8_alpha=session_dump.get("T8_alpha", []),
            t8_beta_l=session_dump.get("T8_betaL", []),
            t8_beta_h=session_dump.get("T8_betaH", []),
            t8_gamma=session_dump.get("T8_gamma", []),
            # FC6
            fc6_theta=session_dump.get("FC6_theta", []),
            fc6_alpha=session_dump.get("FC6_alpha", []),
            fc6_beta_l=session_dump.get("FC6_betaL", []),
            fc6_beta_h=session_dump.get("FC6_betaH", []),
            fc6_gamma=session_dump.get("FC6_gamma", []),
            # F4
            f4_theta=session_dump.get("F4_theta", []),
            f4_alpha=session_dump.get("F4_alpha", []),
            f4_beta_l=session_dump.get("F4_betaL", []),
            f4_beta_h=session_dump.get("F4_betaH", []),
            f4_gamma=session_dump.get("F4_gamma", []),
            # F8
            f8_theta=session_dump.get("F8_theta", []),
            f8_alpha=session_dump.get("F8_alpha", []),
            f8_beta_l=session_dump.get("F8_betaL", []),
            f8_beta_h=session_dump.get("F8_betaH", []),
            f8_gamma=session_dump.get("F8_gamma", []),
            # AF4
            af4_theta=session_dump.get("AF4_theta", []),
            af4_alpha=session_dump.get("AF4_alpha", []),
            af4_beta_l=session_dump.get("AF4_betaL", []),
            af4_beta_h=session_dump.get("AF4_betaH", []),
            af4_gamma=session_dump.get("AF4_gamma", []),
        )

        db.add(eeg_data)
        db.commit()
        db.refresh(eeg_data)
        print(f"Saved EPOC X EEG data (session_id={session_id}) to database.")

    except Exception as e:
        print(f"No data or error encountered: {str(e)}")
