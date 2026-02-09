#!/usr/bin/env python3
"""
Live RealSense D435 inference with Ultralytics YOLO (boxes + seg masks).

Supports USB 2.0 (auto-negotiates resolution) and USB 3.0.

Usage:
  python scripts/realsense_live_seg.py \
    --weights runs/segment/runs/ycb_berkeley_yolo26_seg2/weights/best.pt

Press 'q' or ESC to quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="RealSense live YOLO (boxes + seg masks).")
    p.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/segment/runs/ycb_berkeley_yolo26_seg2/weights/best.pt"),
        help="Path to Ultralytics .pt weights",
    )
    p.add_argument("--device", type=str, default="0", help="YOLO device: 0 (GPU), cpu, etc.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--half", action="store_true", help="FP16 inference")
    p.add_argument("--retina-masks", action="store_true", help="High-res masks (slower)")
    p.add_argument("--window", type=str, default="YOLO RealSense", help="Window title")
    args = p.parse_args()

    # ---- imports ----
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        raise SystemExit(f"pip install opencv-python numpy\n{e}")

    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise SystemExit(f"pip install pyrealsense2\n{e}")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(f"pip install ultralytics\n{e}")

    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    # ---- detect camera ----
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise SystemExit("No RealSense devices found.")

    dev = devs[0]
    serial = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    usb_type = "unknown"
    try:
        usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
    except Exception:
        pass
    print(f"Camera: {name}  serial={serial}  USB={usb_type}")

    # ---- hardware reset (fixes many USB 2.0 streaming issues) ----
    print("Performing hardware reset on camera...")
    dev.hardware_reset()
    print("Waiting for camera to re-enumerate (5s)...")
    time.sleep(5)

    # Re-discover after reset.
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise SystemExit("Camera disappeared after reset. Unplug/replug and retry.")
    dev = devs[0]
    serial = dev.get_info(rs.camera_info.serial_number)

    # ---- try multiple stream configs (USB 2.0 needs lower bandwidth) ----
    stream_configs = [
        # (width, height, fps, format, label)
        (424, 240, 15, rs.format.yuyv, "424x240@15 yuyv"),
        (424, 240, 6, rs.format.yuyv, "424x240@6 yuyv"),
        (640, 480, 6, rs.format.yuyv, "640x480@6 yuyv"),
        (424, 240, 15, rs.format.rgb8, "424x240@15 rgb8"),
        (640, 480, 15, rs.format.rgb8, "640x480@15 rgb8"),
        (None, None, None, None, "auto (SDK default)"),  # fully auto
    ]

    pipeline = None
    profile = None
    active_label = None
    active_fmt = None

    for w, h, fps, fmt, label in stream_configs:
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            if w is not None:
                cfg.enable_stream(rs.stream.color, w, h, fmt, fps)
            else:
                cfg.enable_stream(rs.stream.color)
            print(f"  Trying {label}...", end=" ", flush=True)
            prof = pipe.start(cfg)

            # Actually wait for a frame to confirm it works.
            got_frame = False
            t0 = time.time()
            while time.time() - t0 < 8.0:
                try:
                    frames = pipe.wait_for_frames(5000)
                    cf = frames.get_color_frame()
                    if cf:
                        got_frame = True
                        break
                except RuntimeError:
                    pass

            if got_frame:
                pipeline = pipe
                profile = prof
                active_label = label
                color_profile = prof.get_stream(rs.stream.color).as_video_stream_profile()
                active_fmt = color_profile.format()
                print("OK!")
                break
            else:
                print("configured but no frames.")
                pipe.stop()
        except Exception as e:
            print(f"failed ({e})")

    if pipeline is None or profile is None:
        raise SystemExit(
            "Could not get frames from the RealSense with any configuration.\n"
            "Your camera is on USB 2.1 which has limited bandwidth.\n\n"
            "Possible fixes:\n"
            "  1. Install full librealsense2 (not just pip pyrealsense2):\n"
            "       https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md\n"
            "     This patches the kernel uvcvideo module which is needed for USB 2.0.\n"
            "  2. Find a USB 3.0 port (blue USB-A or USB-C).\n"
            "  3. Try: sudo modprobe uvcvideo\n"
        )

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    actual_w = color_profile.width()
    actual_h = color_profile.height()
    actual_fps = color_profile.fps()
    print(f"Stream active: {active_label} -> {actual_w}x{actual_h}@{actual_fps} format={active_fmt}")

    is_rgb = (active_fmt == rs.format.rgb8)
    is_yuyv = (active_fmt == rs.format.yuyv)

    # ---- load YOLO model ----
    model = YOLO(str(args.weights))
    print(f"Model loaded: {args.weights}")
    print("First frame received — starting live inference. Press 'q' or ESC to quit.")

    # ---- main loop ----
    fps_ema: float | None = None
    alpha = 0.1
    prev = time.time()

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(5000)
            except RuntimeError:
                continue
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            h = color_frame.get_height()
            w = color_frame.get_width()
            
            if is_yuyv:
                # YUYV is 2 bytes per pixel. OpenCV COLOR_YUV2BGR_YUY2 needs (H, W, 2).
                raw_bytes = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                yuv = raw_bytes.reshape((h, w, 2))
                img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUY2)
            else:
                raw = np.asanyarray(color_frame.get_data())
                if is_rgb:
                    img = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                else:
                    img = raw  # bgr or other, pass as-is

            results = model.predict(
                source=img,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                half=args.half,
                retina_masks=args.retina_masks,
                verbose=False,
            )
            r = results[0]
            annotated = r.plot()

            now = time.time()
            inst_fps = 1.0 / max(now - prev, 1e-9)
            prev = now
            fps_ema = inst_fps if fps_ema is None else (1 - alpha) * fps_ema + alpha * inst_fps

            cv2.putText(
                annotated,
                f"FPS: {fps_ema:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(args.window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
