"""Read-only RGB/depth viewer: python -m vision.live_ball."""
import argparse
import asyncio
import json
import queue
import threading
import time
import tkinter as tk

import cv2
import numpy as np
from grpclib.exceptions import StreamTerminatedError
from viam.components.camera import Camera

from ball_tracking import detect_red_ball
from connection import connect


def annotate(color, depth, intrinsics, radius, timestamp, target=None):
    observation = detect_red_ball(color, depth, intrinsics, radius, timestamp)
    display = color.copy()
    target = target or [intrinsics["cx"], intrinsics["cy"]]
    aim = tuple(round(value) for value in target)
    cv2.drawMarker(display, aim, (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
    status = "No single red ball with valid depth detected"
    if observation is not None:
        left, top, right, bottom = observation.bbox
        center = tuple(round(value) for value in observation.pixel)
        cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.drawMarker(display, center, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.line(display, center, aim, (255, 255, 0), 2)
        u, v = observation.pixel
        x, y, z = observation.xyz
        status = (f"Box midpoint: ({u:.1f}, {v:.1f}) px | surface depth: {observation.depth_mm:.1f} mm\n"
                  f"Ball center CAMERA XYZ: ({x:.1f}, {y:.1f}, {z:.1f}) mm\n"
                  f"Alignment error: du={u-target[0]:+.1f}, dv={v-target[1]:+.1f} px "
                  "(right/down positive)")
    valid = np.isfinite(depth) & (depth > 0)
    scaled = np.uint8(np.clip(np.nan_to_num(depth, nan=0) / 3000 * 255, 0, 255))
    depth_view = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    depth_view[~valid] = 0
    if observation is not None:
        cv2.drawMarker(depth_view, center, (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
    return np.hstack((display, depth_view)), status


async def stream(config, frames, stop):
    def publish(item):
        if frames.full():
            try:
                frames.get_nowait()
            except queue.Empty:
                pass
        frames.put_nowait(item)

    retry_delay = 1
    while not stop.is_set():
        machine = None
        try:
            machine = await asyncio.wait_for(connect(), timeout=20)
            camera = Camera.from_robot(machine, config.get("camera", "cam"))
            intrinsics = config.get("intrinsics")
            if intrinsics is None:
                properties = await camera.get_properties(timeout=5)
                p = properties.intrinsic_parameters
                intrinsics = dict(width=p.width_px, height=p.height_px, fx=p.focal_x_px,
                                  fy=p.focal_y_px, cx=p.center_x_px, cy=p.center_y_px)
            previous = None
            while not stop.is_set():
                try:
                    images, metadata = await camera.get_images(timeout=5)
                    sources = {image.name: image for image in images}
                    color_name, depth_name = config.get("color_source", "color"), config.get("depth_source", "depth")
                    if color_name not in sources or depth_name not in sources:
                        raise ValueError(f"Available camera sources: {list(sources)}; configure color_source/depth_source")
                    color = cv2.imdecode(np.frombuffer(sources[color_name].data, np.uint8), cv2.IMREAD_COLOR)
                    depth = np.asarray(sources[depth_name].bytes_to_depth_array(), dtype=float)
                    timestamp = metadata.captured_at.seconds + metadata.captured_at.nanos / 1e9
                    if timestamp <= (previous or 0) or not 0 <= time.time()-timestamp <= config.get("max_age_s", 0.25):
                        raise ValueError("Stale/repeated frame: check camera stream and clock synchronization")
                    previous = timestamp
                    image, status = annotate(color, depth, intrinsics, config.get("ball_radius_mm", 20), timestamp,
                                             config.get("alignment_pixel"))
                    if image.shape[1] > 1400:
                        image = cv2.resize(image, (1400, round(image.shape[0]*1400/image.shape[1])))
                    ok, png = cv2.imencode(".png", image)
                    if not ok:
                        raise ValueError("Could not encode preview")
                    publish((png.tobytes(), status))
                    retry_delay = 1
                except (StreamTerminatedError, ConnectionError, OSError, asyncio.TimeoutError):
                    # A terminated stream cannot be repaired by reusing its client.
                    raise
                except Exception as error:
                    publish((None, f"{type(error).__name__}: {error}"))
                    await asyncio.sleep(0.3)
                await asyncio.sleep(0.05)
        except Exception as error:
            publish((None, f"{type(error).__name__}: {error}\n"
                     f"Reconnecting in {retry_delay}s. Check that the Viam machine is Live "
                     "and cam works in CONTROL."))
        finally:
            if machine is not None:
                try:
                    await asyncio.wait_for(machine.close(), timeout=3)
                except Exception:
                    pass
        for _ in range(retry_delay * 10):
            if stop.is_set():
                return
            await asyncio.sleep(0.1)
        retry_delay = min(retry_delay * 2, 10)


async def probe(camera_name):
    """Read-only connection test; never prints credentials or image contents."""
    machine = await asyncio.wait_for(connect(), timeout=20)
    try:
        print("Connected. Cameras:", [r.name for r in machine.resource_names if r.subtype == "camera"], flush=True)
        camera = Camera.from_robot(machine, camera_name)
        properties = await camera.get_properties(timeout=10)
        print("Intrinsics:", properties.intrinsic_parameters, flush=True)
        for index in range(3):
            images, metadata = await camera.get_images(timeout=10)
            print("Frame", index+1, [(im.name, str(im.mime_type), len(im.data)) for im in images], flush=True)
            print("Capture timestamp present:", bool(metadata.captured_at.seconds), flush=True)
            await asyncio.sleep(0.2)
    finally:
        await machine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Measured camera settings; otherwise use camera-reported intrinsics")
    parser.add_argument("--camera", default="cam")
    parser.add_argument("--radius-mm", type=float, default=20)
    parser.add_argument("--probe", action="store_true", help="Check connection and image sources without a GUI")
    args = parser.parse_args()
    if args.probe:
        asyncio.run(probe(args.camera))
        return
    config = {"camera": args.camera, "ball_radius_mm": args.radius_mm}
    if args.config:
        with open(args.config, encoding="utf-8") as file:
            config.update(json.load(file))
    window = tk.Tk()
    window.title("Viam red ball | RGB + depth | observation only")
    tk.Label(window, text="RGB: green box / yellow midpoint / cyan alignment target     |     Aligned depth: 0–3000 mm").pack()
    display = tk.Label(window, text="Connecting to Viam camera…", width=100, height=25)
    display.pack()
    status = tk.StringVar(value="Connecting…")
    tk.Label(window, textvariable=status, font=("Consolas", 11), justify="left").pack(padx=12, pady=8)
    tk.Label(window, text=f"Camera coordinates: X right, Y down, Z forward. Radius: {config['ball_radius_mm']} mm.\n"
             "Confirm aligned COLOR intrinsics. Pixel alignment alone is not gripper alignment; wrist/tool calibration is required.").pack(pady=6)
    frames, stop = queue.Queue(maxsize=1), threading.Event()
    worker = threading.Thread(target=lambda: asyncio.run(stream(config, frames, stop)), daemon=True)
    worker.start()

    def refresh():
        try:
            png, message = frames.get_nowait()
            status.set(message)
            if png is None:
                display.configure(image="", text="No current valid frame", width=100, height=25)
                display.image = None
            else:
                photo = tk.PhotoImage(data=png)
                display.configure(image=photo, text="", width=0, height=0)
                display.image = photo
        except queue.Empty:
            pass
        window.after(50, refresh)

    def close():
        stop.set()
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", close)
    window.bind("<Escape>", lambda event: close())
    refresh()
    window.mainloop()


if __name__ == "__main__":
    main()
