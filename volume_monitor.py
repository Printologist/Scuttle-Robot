"""
Monitors speaker audio output volume and sends values over serial
to CircuitPython board to drive NeoPixel LEDs.

Requires: pip install pyserial sounddevice numpy --break-system-packages
"""

import serial
import sounddevice as sd
import numpy as np
import time

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
MAX_VOLUME = 3000  # tune this up or down based on your speaker volume


def get_volume(indata):
    """Calculate RMS volume from audio block."""
    samples = indata[:, 0].astype(np.float32)
    rms = np.sqrt(np.mean(samples ** 2))
    return rms


def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"Could not open serial port: {e}")
        return

    print("Monitoring speaker volume...")

    def audio_callback(indata, frames, time_info, status):
        volume = get_volume(indata)
        normalized = min(volume / MAX_VOLUME, 1.0)
        try:
            ser.write(f"v:{normalized:.2f}\n".encode())
        except Exception as e:
            print(f"Serial write error: {e}")

    # List available devices to help find the right one
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    print("\nUsing default input device. If wrong, set device= in InputStream\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        callback=audio_callback
    ):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            ser.close()


if __name__ == "__main__":
    main()
