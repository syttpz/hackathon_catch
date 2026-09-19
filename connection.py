import asyncio
import os

from dotenv import load_dotenv
from viam.robot.client import RobotClient

load_dotenv()


async def connect():
    api_key = os.environ["VIAM_API_KEY"]
    api_key_id = os.environ["VIAM_API_KEY_ID"]
    address = os.environ["VIAM_ADDRESS"]

    options = RobotClient.Options.with_api_key(
        api_key=api_key,
        api_key_id=api_key_id,
    )
    timeout = float(os.getenv("VIAM_DIAL_TIMEOUT_S", "60"))
    if options.dial_options is not None:
        options.dial_options.timeout = timeout
        options.dial_options.initial_connection_attempt_timeout = timeout
        options.dial_options.initial_connection_attempts = 5

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            print(f"Connecting to {address} (attempt {attempt}/3, timeout {timeout:.0f}s)...", flush=True)
            return await RobotClient.at_address(address, options)
        except Exception as error:
            last_error = error
            print(f"Dial failed: {error}", flush=True)
            await asyncio.sleep(2 * attempt)
    raise RuntimeError(
        f"Could not reach {address}. The machine may be offline, or the previous "
        "WebRTC proxy from another script may still be tearing down. Wait a few "
        "seconds and retry, and confirm the machine is live in the Viam app. "
        f"Last error: {last_error}"
    ) from last_error
