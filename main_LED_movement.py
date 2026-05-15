"""
Voice assistant using OpenAI Whisper, GPT-4o, and OpenAI TTS.
With camera vision support via Viam and NeoPixel volume display.

Requires: pip install viam-sdk openai pyserial --break-system-packages
"""

import asyncio
import os
import wave
import base64
import serial
from io import BytesIO

from viam.robot.client import RobotClient
from viam.components.audio_in import AudioIn, AudioCodec
from viam.components.audio_out import AudioOut, AudioInfo
from viam.components.camera import Camera
from viam.components.base import Base
from viam.media.video import CameraMimeType
from openai import OpenAI


SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200


def get_serial():
    """Try to open serial connection to CircuitPython board."""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to NeoPixels on {SERIAL_PORT}")
        return ser
    except Exception as e:
        print(f"Could not connect to NeoPixels: {e}")
        return None


def send_volume(ser, volume: float):
    """Send volume value (0.0 to 1.0) over serial."""
    if ser:
        try:
            ser.write(f"v:{volume:.2f}\n".encode())
        except Exception:
            pass


def get_volume_chunks(mp3_data: bytes, chunk_ms: int = 100):
    """Decode MP3 and return list of (volume, duration_seconds) per chunk."""
    from pydub import AudioSegment
    import numpy as np

    audio = AudioSegment.from_mp3(BytesIO(mp3_data))
    audio = audio.set_channels(1)  # mono

    chunks = []
    for i in range(0, len(audio), chunk_ms):
        chunk = audio[i:i + chunk_ms]
        samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
        if len(samples) == 0:
            continue
        rms = np.sqrt(np.mean(samples ** 2))
        # Normalize: max RMS for 16-bit audio is 32768
        normalized = min(rms / 32768.0 * 3.0, 1.0)  # *3 to boost sensitivity
        chunks.append((normalized, chunk_ms / 1000.0))

    return chunks


async def animate_leds_realtime(ser, mp3_data: bytes):
    """Send volume updates in real time as audio plays."""
    try:
        chunks = get_volume_chunks(mp3_data)
        for volume, duration in chunks:
            send_volume(ser, volume)
            await asyncio.sleep(duration)
        send_volume(ser, 0.0)
    except Exception as e:
        print(f"LED animation error: {e}")
        send_volume(ser, 0.0)


class OpenAIVoiceAssistant:
    """Voice assistant powered by OpenAI services with camera vision."""

    def __init__(
        self,
        robot: RobotClient,
        filter_name: str = "wake-word",
        audioout_name: str = "speaker",
        camera_name: str = "camera",
        base_name: str = "base",
        ser=None,
    ):
        self.robot = robot
        self.filter_name = filter_name
        self.audioout_name = audioout_name
        self.camera_name = camera_name
        self.base_name = base_name
        self.filter = None
        self.audioout = None
        self.camera = None
        self.base = None
        self.ser = ser

        self.client = OpenAI(api_key='YOUR_OPENAI_KEY_HERE')
        self.system_prompt = (
            "You are a helpful voice assistant on a robot. "
            "Keep responses concise and conversational. "
            "When the user asks you to move, respond with ONLY one of these exact words: "
            "MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, STOP. "
            "For all other requests, respond normally."
        )
        self.chat_history = []

        # Keywords that trigger a camera capture
        self.vision_keywords = [
            "see", "look", "what is", "what's", "describe",
            "in front", "around", "camera", "show", "watch"
        ]

        # Keywords that trigger movement
        self.movement_keywords = [
            "forward", "backward", "back", "turn left", "turn right", "stop"
        ]

    async def start(self):
        self.filter = AudioIn.from_robot(self.robot, self.filter_name)
        self.audioout = AudioOut.from_robot(self.robot, self.audioout_name)
        self.camera = Camera.from_robot(self.robot, self.camera_name)
        self.base = Base.from_robot(self.robot, self.base_name)
        print(f"Connected to wake-word filter: {self.filter_name}")
        print(f"Connected to speaker: {self.audioout_name}")
        print(f"Connected to camera: {self.camera_name}")
        print(f"Connected to base: {self.base_name}")

    def is_movement_request(self, text: str) -> bool:
        """Check if the user is asking the robot to move."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.movement_keywords)

    async def execute_movement(self, command: str):
        """Execute a movement command on the base."""
        try:
            command = command.strip().upper()
            if command == "MOVE_FORWARD":
                print("Moving forward 500mm...")
                await self.base.move_straight(distance=500, velocity=300)
            elif command == "MOVE_BACKWARD":
                print("Moving backward 500mm...")
                await self.base.move_straight(distance=-500, velocity=300)
            elif command == "TURN_LEFT":
                print("Turning left 45 degrees...")
                await self.base.spin(angle=45, velocity=60)
            elif command == "TURN_RIGHT":
                print("Turning right 45 degrees...")
                await self.base.spin(angle=-45, velocity=60)
            elif command == "STOP":
                print("Stopping...")
                await self.base.stop()
        except Exception as e:
            print(f"Movement error: {e}")

    def is_vision_request(self, text: str) -> bool:
        """Check if the user is asking about what the robot sees."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.vision_keywords)

    async def get_camera_image(self) -> str | None:
        """Capture a frame from the camera and return as base64."""
        try:
            result = await self.camera.get_images()
            named_images = result[0]
            image = named_images[0]
            image_bytes = image.data
            print(f"Image captured, size: {len(image_bytes)} bytes")
            return base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def speech_to_text(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Convert audio to text using Whisper."""
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)

        wav_buffer.seek(0)
        wav_buffer.name = "audio.wav"
        response = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_buffer,
        )
        return response.text

    async def get_response(self, user_text: str) -> str:
        """Generate response using GPT-4o, with vision and movement if needed."""
        if not user_text:
            return "I didn't catch that."

        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.chat_history)

            if self.is_vision_request(user_text):
                print("Vision request detected, capturing image...")
                image_b64 = await self.get_camera_image()

                if image_b64:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": user_text
                            }
                        ]
                    })
                else:
                    messages.append({"role": "user", "content": user_text})
            else:
                messages.append({"role": "user", "content": user_text})

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )

            assistant_message = response.choices[0].message.content

            # Check if response is a movement command
            movement_commands = ["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
            if assistant_message.strip().upper() in movement_commands:
                await self.execute_movement(assistant_message)
                return "OK, moving now."

            self.chat_history.append({"role": "user", "content": user_text})
            self.chat_history.append({"role": "assistant", "content": assistant_message})

            return assistant_message
        except Exception as e:
            print(f"Error getting GPT response: {e}")
            return "Sorry, I had trouble processing that."

    async def speak(self, text: str):
        """Text to speech using OpenAI TTS, with LED volume animation."""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
            )
            mp3_data = response.content

            # Play audio and animate LEDs concurrently in real time
            audio_info = AudioInfo(codec=AudioCodec.MP3)
            await asyncio.gather(
                self.audioout.play(mp3_data, audio_info),
                animate_leds_realtime(self.ser, mp3_data)
            )
        except Exception as e:
            print(f"Error in text to speech: {e}")

    async def run(self):
        """Continuously listen and respond."""
        print("Listening for wake word 'robot'...")

        while True:
            try:
                audio_stream = await self.filter.get_audio("pcm16", 0, 0)
            except Exception as e:
                print(f"Error starting audio stream: {e}, retrying...")
                await asyncio.sleep(1)
                continue

            try:
                segment = bytearray()

                async for chunk in audio_stream:
                    audio_data = chunk.audio.audio_data

                    if len(audio_data) == 0:
                        if segment:
                            print(f"\nWake word detected! Processing {len(segment)} bytes...")
                            try:
                                user_text = self.speech_to_text(bytes(segment))
                                if user_text:
                                    print(f"You: {user_text}")
                                    response_text = await self.get_response(user_text)
                                    print(f"Bot: {response_text}")
                                    await self.speak(response_text)
                                else:
                                    print("No speech recognized")
                            except Exception as e:
                                print(f"Error processing speech: {e}")

                            segment.clear()
                            print("Listening for next wake word...\n")
                    else:
                        segment.extend(audio_data)

            except KeyboardInterrupt:
                print("\n\nStopping...")
                return
            except Exception as e:
                print(f"Stream disconnected: {e}, reconnecting...")
                await asyncio.sleep(1)
                continue


async def main():
    ser = get_serial()

    opts = RobotClient.Options.with_api_key(
        api_key='',
        api_key_id=''
    )
    robot = await RobotClient.at_address('', opts)

    try:
        assistant = OpenAIVoiceAssistant(robot, "wake-word", "speaker", "camera", "base", ser)
        await assistant.start()
        await assistant.run()
    finally:
        await robot.close()
        if ser:
            ser.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nStopped by user")
