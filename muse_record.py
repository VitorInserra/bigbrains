import asyncio, time
import multiprocessing as mp
from muselsl import list_muses, stream

def stream_muse():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        try:
            muses = list_muses()
            if muses:
                print(f"Connecting to Muse: {muses[0]['name']}...")
                stream(muses[0]["address"])
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

