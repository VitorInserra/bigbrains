import asyncio
import websockets
import json
from datetime import datetime
import time
import ssl
import os
import uuid
import threading
import pandas as pd
from db_handling.EpocXData import insert_eeg_db
from sqlalchemy.orm import Session
from fastapi import FastAPI, Depends
from db import get_db
from dotenv import load_dotenv

CORTEX_URL = "wss://127.0.0.1:6868"

load_dotenv()

CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")

ssl_context = ssl._create_unverified_context()


def create_payload(method, params, request_id):
    """
    Create a JSON-RPC 2.0 payload.
    """
    return {"jsonrpc": "2.0", "method": method, "params": params, "id": request_id}


async def send_json_rpc(websocket, method, params, request_id):
    """
    Utility function to send a JSON-RPC 2.0 request to the Cortex API.
    """
    payload = create_payload(method, params, request_id)
    await websocket.send(json.dumps(payload))
    response = await websocket.recv()
    return json.loads(response)


async def request_access(websocket, client_id, client_secret):
    print("==== 1) requestAccess ====")
    return await send_json_rpc(
        websocket,
        "requestAccess",
        {"clientId": client_id, "clientSecret": client_secret},
        request_id=1,
    )


async def control_device(websocket, command, request_id, headset_id=None):
    print(f"\n==== controlDevice ({command}) ====")
    params = {"command": command}
    if headset_id:
        params["headset"] = headset_id
    return await send_json_rpc(websocket, "controlDevice", params, request_id)


async def query_headsets(websocket):
    print("\n==== 3) queryHeadsets ====")
    return await send_json_rpc(websocket, "queryHeadsets", {}, request_id=3)


async def authorize(websocket, client_id, client_secret):
    print("\n==== 5) authorize ====")
    return await send_json_rpc(
        websocket,
        "authorize",
        {"clientId": client_id, "clientSecret": client_secret, "debit": 1},
        request_id=5,
    )


async def create_session(websocket, cortex_token, headset_id):
    print("\n==== 6) createSession ====")
    return await send_json_rpc(
        websocket,
        "createSession",
        {"cortexToken": cortex_token, "headset": headset_id, "status": "open"},
        request_id=6,
    )


async def subscribe_to_streams(websocket, cortex_token, session_id):
    print("\n==== subscribe (band power) ====")
    return await send_json_rpc(
        websocket,
        "subscribe",
        {"cortexToken": cortex_token, "session": session_id, "streams": ["pow"]},
        request_id=7,
    )

pow_data_batch = []
async def handle_incoming_data(websocket):
    batch_size = -1
    while True:
        data_msg = await websocket.recv()
        data = json.loads(data_msg)
        if "pow" in data:
            channel_values = data["pow"]
            row = {
                "timestamp": data["time"],
                "AF3": channel_values[0],
                "F7": channel_values[1],
                "F3": channel_values[2],
                "FC5": channel_values[3],
                "T7": channel_values[4],
                "P7": channel_values[5],
                "O1": channel_values[6],
                "O2": channel_values[7],
                "P8": channel_values[8],
                "T8": channel_values[9],
                "FC6": channel_values[10],
                "F4": channel_values[11],
                "F8": channel_values[12],
                "AF4": channel_values[13],
            }

            pow_data_batch.append(row)

            if len(pow_data_batch) == batch_size:
                df = pd.DataFrame(pow_data_batch)
                pow_data_batch.clear()




async def get_detection_info(websocket, cortex_token, session_id):
        await send_json_rpc(
            websocket,
            "subscribe",
            {"cortexToken": cortex_token, "session": session_id, "streams": ["fac"]},
            request_id=7,
        )

        while True:
            response = await websocket.recv()

            timestamp = datetime.now().isoformat()

            print(f"Timestamp: {timestamp}")
            print(f"getDetectionInfo Response:\n{response}")
            time.sleep(1)

async def main():
    async with websockets.connect(CORTEX_URL, ssl=ssl_context) as websocket:

        resp = await request_access(websocket, CLIENT_ID, CLIENT_SECRET)
        print("requestAccess response:", resp)

        await control_device(websocket, "refresh", request_id=2)

        time.sleep(1)

        resp = await query_headsets(websocket)
        print("queryHeadsets response:", resp)
        headsets = resp.get("result", [])
        if not headsets:
            print("No headsets found. Make sure your device is on and in range.")
            return

        headset_id = headsets[0]["id"]
        print(f"Using headset_id: {headset_id}")

        await control_device(websocket, "connect", request_id=4, headset_id=headset_id)

        resp = await authorize(websocket, CLIENT_ID, CLIENT_SECRET)
        print("authorize response:", resp)
        cortex_token = resp["result"]["cortexToken"]
        print(f"Cortex Token: {cortex_token}")

        resp = await create_session(websocket, cortex_token, headset_id)
        print("createSession response:", resp)
        session_id = resp["result"]["id"]

        # await subscribe_to_streams(websocket, cortex_token, session_id)
        # await handle_incoming_data(websocket)

        await get_detection_info(websocket, cortex_token, session_id)


if __name__ == "__main__":
    asyncio.run(main())
