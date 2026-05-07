from __future__ import annotations

import asyncio
import json
import websockets


async def main() -> None:
    uri = "ws://localhost:8000/audio/whisper?interval_sec=3"
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            print(json.loads(message))


if __name__ == "__main__":
    asyncio.run(main())
