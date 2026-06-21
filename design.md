# ReSpeaker 4 Mic Array — Design Notes

## Device

**Seeed Technology ReSpeaker 4 Mic Array (UAC1.0)**
- USB VID:PID `2886:0018`, bcdDevice 2.00
- ALSA: `card 2`, `ArrayUAC10`, device name `ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:2,0)`

### USB Interface Map

| # | Name | Class | Notes |
|---|------|-------|-------|
| 0 | AudioControl | 0x01/0x01 | UAC1.0 control; 2 terminals each direction |
| 1 | AudioStreaming OUT | 0x01/0x02 | Playback: 2ch, 24-bit, 16 kHz |
| 2 | AudioStreaming IN  | 0x01/0x02 | Capture: **6ch, 16-bit, 16 kHz** — the mic data |
| 3 | SEEED Control | 0xff/0xff/0xff | Vendor-specific, no endpoints |
| 4 | SEEED DFU | 0xfe/0x01 | Device firmware update, not relevant |

### AudioControl topology (interface 0)

```
INPUT_TERMINAL 1  (USB Streaming, 2ch)  →  OUTPUT_TERMINAL 6  (Speaker)
INPUT_TERMINAL 2  (Microphone, 6ch)     →  OUTPUT_TERMINAL 7  (USB Streaming → interface 2)
```

Notably: **no Extension Units** in the AudioControl descriptor. DOA is not exposed
via standard UAC1.0 extension unit control selectors.

### Capture stream (interface 2, alt 1)

- Isochronous IN, endpoint `0x82`, max packet 204 bytes
- 6 channels × 2 bytes × 16 kHz = 192 kB/s
- Channel layout (from Seeed documentation): ch0–3 = raw mics (square arrangement),
  ch4 = AEC-processed output, ch5 = unused/silence

---

## What Works

### Audio capture

`sounddevice` wraps ALSA and opens the device cleanly. The 6-channel `float32`
stream is accessible at `hw:2,0` (ALSA index 4 in sounddevice).

```python
sd.InputStream(device=4, channels=6, samplerate=16000, dtype="float32", ...)
```

This is implemented and working in `mic_array.py`.

### udev access

A udev rule (`99-respeaker.rules`) grants the `plugdev` group access to the USB
device so `pyusb` can open it without `sudo`:

```
SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0018", MODE="0660", GROUP="plugdev"
```

---

## Onboard DOA via USB Control Transfer

### Initial failure

The XMOS XVF3000 chip computes DOA onboard and exposes it as a readable parameter
via vendor USB control transfers. Initial attempts were based on training-data
knowledge of Seeed's `tuning.py`, which documented this pattern:

```python
dev.ctrl_transfer(
    CTRL_IN | CTRL_TYPE_VENDOR | CTRL_RECIPIENT_DEVICE,
    bRequest=0, wValue=param_id, wIndex=0x1C, wLength=4
)
```

All 18 combinations of `bRequest`, `wValue`, `wIndex`, `wLength`, and recipient
(`DEVICE` vs `INTERFACE`) returned **`[Errno 32] Pipe error`** (USB STALL on EP0).

The working hypothesis was that `tuning.py` targets the **ReSpeaker USB Mic Array
v2.0** (`0x2886:0x0007/0x0008`) and its protocol doesn't apply to the UAC1.0
variant (`0x0018`). Usbmon capture was being considered as the next step.

### Resolution

A web search surfaced the canonical source:
- **`respeaker/usb_4_mic_array` GitHub wiki — "USB Control Protocol"**
  https://github.com/respeaker/usb_4_mic_array/wiki/USB-Control-Protocol
- **`tuning.py` source** (the version specific to this device, `0x0018`):
  https://github.com/respeaker/usb_4_mic_array/blob/master/tuning.py

The actual read protocol encodes read/write and type information into `wValue`,
and puts the parameter ID in `wIndex` — the opposite of what was assumed:

| Field | Wrong assumption | Correct value |
|-------|-----------------|---------------|
| `wValue` | `param_id` (21) | `0xC0` — `0x80` (read flag) \| `0x40` (int type) |
| `wIndex` | `0x1C` | `param_id` (21 for `DOAANGLE`) |
| `wLength` | `4` | `8` (two 32-bit words returned) |
| Result | — | `data[0:4]` as little-endian int32 |

Correct call:

```python
data = dev.ctrl_transfer(
    CTRL_IN | CTRL_TYPE_VENDOR | CTRL_RECIPIENT_DEVICE,
    0, 0xC0, 21, 8, timeout=1000
)
angle = int.from_bytes(data[0:4], 'little', signed=True)  # 0-359°
```

This is implemented and working in `mic_array.py`. Live reads confirm the angle
tracks ambient sound direction at ~100 ms polling rate.

---

## File Map

| File | Purpose |
|------|---------|
| `mic_array.py` | Main app: audio capture + onboard DOA display |
| `99-respeaker.rules` | udev rule for non-root USB access |
| `design.md` | This file |

---

## References

| Resource | URL | Used for |
|----------|-----|---------|
| `usb_4_mic_array` GitHub repo | https://github.com/respeaker/usb_4_mic_array | Confirmed this is the correct repo for `0x0018` |
| USB Control Protocol wiki | https://github.com/respeaker/usb_4_mic_array/wiki/USB-Control-Protocol | Correct `wValue`/`wIndex`/`wLength` encoding |
| `tuning.py` source | https://github.com/respeaker/usb_4_mic_array/blob/master/tuning.py | `DOAANGLE` param ID (21), read method details |
| Seeed Studio wiki (UAC1.0) | https://wiki.seeedstudio.com/ReSpeaker-USB-Mic-Array/ | Device overview, channel layout |
