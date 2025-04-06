import websockets
import asyncio

async def test():
    uri = "ws://localhost:8000"
    async with websockets.connect(uri) as ws:
        try:
            print(await ws.recv())
        except Exception as e:
            print(f"Error: {e}")

asyncio.run(test())
