#!/usr/bin/env python3
"""
ReSpeaker 4 Mic Array — audio capture + onboard DOA
Audio:  sounddevice -> ALSA card "ReSpeaker 4 Mic Array (UAC1.0)"
DOA:    pyusb vendor control transfer to the XMOS XVF3000 chip
"""

import threading
import time

import numpy as np
import sounddevice as sd
import usb.core
import usb.util

# ── USB device identifiers ────────────────────────────────────────────────────
VENDOR_ID  = 0x2886
PRODUCT_ID = 0x0018

# XMOS parameter IDs (from Seeed usb_4_mic_array reference implementation)
PARAM_DOA_ANGLE = 21   # 0-359 degrees

# Control-transfer constants
CTRL_IN  = usb.util.CTRL_IN  | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE
CTRL_OUT = usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE

# ── Audio settings ────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
CHANNELS      = 6       # UAC1.0 exposes 6 channels (4 mics + 2 processed)
BLOCK_FRAMES  = 1024
DEVICE_NAME   = "ReSpeaker 4 Mic Array"


class ReSpeaker:
    def __init__(self):
        self.dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
        if self.dev is None:
            raise RuntimeError("ReSpeaker not found. Check USB connection.")

    def read_int_param(self, param_id: int) -> int:
        """Read an integer parameter from the XMOS chip.

        Protocol (from respeaker/usb_4_mic_array):
          wValue = 0x80 (read) | 0x40 (int type) = 0xC0
          wIndex = param_id
          wLength = 8  (two 32-bit words; value is in bytes [0:4])
        """
        data = self.dev.ctrl_transfer(CTRL_IN, 0, 0xC0, param_id, 8, timeout=1000)
        return int.from_bytes(data[0:4], 'little', signed=True)

    def doa_angle(self) -> int:
        """Return Direction of Arrival angle in degrees (0-359)."""
        return self.read_int_param(PARAM_DOA_ANGLE)


def find_device_index(name: str) -> int:
    """Return the sounddevice input index matching the given device name substring."""
    for i, dev in enumerate(sd.query_devices()):
        if name.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i
    raise RuntimeError(f"Audio device '{name}' not found. Available:\n{sd.query_devices()}")


def rms_db(block: np.ndarray) -> float:
    """RMS level of a block in dBFS."""
    rms = np.sqrt(np.mean(block ** 2))
    if rms < 1e-10:
        return -100.0
    return 20 * np.log10(rms)


def bar(value: float, lo: float, hi: float, width: int = 20) -> str:
    filled = int((value - lo) / (hi - lo) * width)
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main():
    print("Connecting to ReSpeaker…")
    mic = ReSpeaker()
    print("USB device found.")

    dev_idx = find_device_index(DEVICE_NAME)
    dev_info = sd.query_devices(dev_idx)
    print(f"Audio device: {dev_info['name']} (index {dev_idx})")
    print(f"  Channels: {CHANNELS}, Rate: {SAMPLE_RATE} Hz\n")
    print("Press Ctrl-C to stop.\n")

    latest_block = np.zeros(BLOCK_FRAMES, dtype=np.float32)
    lock = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
        nonlocal latest_block
        if status:
            print(f"[audio status] {status}")
        with lock:
            # channel 0 is the first raw mic
            latest_block = indata[:, 0].copy()

    with sd.InputStream(
        device=dev_idx,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_FRAMES,
        dtype="float32",
        callback=audio_callback,
    ):
        try:
            while True:
                with lock:
                    block = latest_block.copy()

                doa   = mic.doa_angle()
                level = rms_db(block)

                level_bar = bar(level, -60, 0, width=30)
                # simple compass arrow based on 8 sectors
                compass = " N  NE  E  SE  S  SW  W  NW".split()
                sector  = compass[round(doa / 45) % 8]

                print(
                    f"\rDOA: {doa:>3}° {sector:<2}  |  "
                    f"Level ch0: {level:>6.1f} dBFS  {level_bar}",
                    end="",
                    flush=True,
                )
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
