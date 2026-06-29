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
import math
import struct
from io import BytesIO

from viam.robot.client import RobotClient
from viam.components.audio_in import AudioIn, AudioCodec
from viam.components.audio_out import AudioOut, AudioInfo
from viam.components.camera import Camera
from viam.components.base import Base
from viam.media.video import CameraMimeType
from viam.proto.common import PointCloudObject
from viam.services.vision import VisionClient
from openai import OpenAI


SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
OBSTACLE_DISTANCE_MM = 50  # stop if anything closer than 1 foot


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
        lidar_name: str = "Lidar",
        vision_name: str = "vision-1",
        ser=None,
    ):
        self.robot = robot
        self.filter_name = filter_name
        self.audioout_name = audioout_name
        self.camera_name = camera_name
        self.base_name = base_name
        self.lidar_name = lidar_name
        self.vision_name = vision_name
        self.filter = None
        self.audioout = None
        self.camera = None
        self.base = None
        self.lidar = None
        self.vision = None
        self.ser = ser
        self.following = False  # follow mode flag

        self.client = OpenAI(api_key='')
        self.system_prompt = (
            "You are a helpful voice assistant on a robot. "
            "Keep responses concise and conversational. "
            "You have the following hardware: a camera, a LIDAR sensor for distance measurement, "
            "a microphone, a speaker, and wheels for movement. "
            "When the user asks you to move, dance, follow, stop following, or get out of the way, "
            "respond with ONLY one of these exact words: "
            "MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, STOP, DANCE, FOLLOW, STOP_FOLLOWING, GET_OUT_OF_WAY. "
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
            "forward", "backward", "back", "turn left", "turn right", "stop",
            "get out of the way", "out of the way", "move out"
        ]

    async def start(self):
        self.filter = AudioIn.from_robot(self.robot, self.filter_name)
        self.audioout = AudioOut.from_robot(self.robot, self.audioout_name)
        self.camera = Camera.from_robot(self.robot, self.camera_name)
        self.base = Base.from_robot(self.robot, self.base_name)
        self.lidar = Camera.from_robot(self.robot, self.lidar_name)
        self.vision = VisionClient.from_robot(self.robot, self.vision_name)
        print(f"Connected to wake-word filter: {self.filter_name}")
        print(f"Connected to speaker: {self.audioout_name}")
        print(f"Connected to camera: {self.camera_name}")
        print(f"Connected to base: {self.base_name}")
        print(f"Connected to LIDAR: {self.lidar_name}")
        print(f"Connected to vision: {self.vision_name}")

    def is_movement_request(self, text: str) -> bool:
        """Check if the user is asking the robot to move."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.movement_keywords)

    async def get_nearest_distance_mm(self) -> float | None:
        """Get the nearest obstacle distance in mm from LIDAR point cloud."""
        try:
            pc, _ = await self.lidar.get_point_cloud()
            # PCD binary format: each point is 3 floats (x, y, z) in meters
            num_floats = len(pc) // 4
            points = struct.unpack(f"{num_floats}f", pc[:num_floats * 4])
            min_dist = float("inf")
            for i in range(0, len(points) - 2, 4):  # x, y, z, intensity
                x, y, z = points[i], points[i+1], points[i+2]
                dist = math.sqrt(x**2 + y**2) * 1000  # meters to mm, ignore z
                if dist > 0 and dist < min_dist:
                    min_dist = dist
            return min_dist if min_dist != float("inf") else None
        except Exception as e:
            print(f"LIDAR error: {e}")
            return None

    async def is_obstacle_ahead(self) -> bool:
        """Check if there's an obstacle within the safety distance."""
        dist = await self.get_nearest_distance_mm()
        if dist is None:
            return False
        print(f"Nearest obstacle: {dist:.0f}mm")
        return dist < OBSTACLE_DISTANCE_MM

    async def execute_movement(self, command: str):
        """Execute a movement command on the base with obstacle avoidance."""
        try:
            command = command.strip().upper()
            if command == "MOVE_FORWARD":
                print("Moving forward 500mm...")
                if await self.is_obstacle_ahead():
                    print("Obstacle detected! Stopping.")
                    return "obstacle"
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
            elif command == "GET_OUT_OF_WAY":
                print("Getting out of the way...")
                await self.base.spin(angle=90, velocity=60)
                await self.base.move_straight(distance=500, velocity=300)
                await self.base.spin(angle=-90, velocity=60)
            elif command == "DANCE":
                print("Do-si-do! Starting square dance...")
                # Promenade forward
                await self.base.move_straight(distance=500, velocity=300)
                await asyncio.sleep(0.5)
                # Swing your partner - spin right
                await self.base.spin(angle=-90, velocity=80)
                await asyncio.sleep(0.5)
                # Circle left
                await self.base.spin(angle=360, velocity=60)
                await asyncio.sleep(0.5)
                # Promenade back
                await self.base.move_straight(distance=-500, velocity=300)
                await asyncio.sleep(0.5)
                # Swing your partner other way
                await self.base.spin(angle=90, velocity=80)
                await asyncio.sleep(0.5)
                # Do-si-do - forward and back
                await self.base.move_straight(distance=300, velocity=300)
                await asyncio.sleep(0.3)
                await self.base.move_straight(distance=-300, velocity=300)
                await asyncio.sleep(0.3)
                # Final spin
                await self.base.spin(angle=360, velocity=80)
                await self.base.stop()
                print("Dance complete!")
            return "ok"
        except Exception as e:
            print(f"Movement error: {e}")
            return "error"

    async def follow_loop(self):
        """Continuously track and follow a person using vision service."""
        print("Follow mode started...")
        FRAME_WIDTH = 640  # adjust if your camera resolution is different
        CENTER_ZONE = 100  # pixels from center considered centered
        FOLLOW_DISTANCE_MM = 600  # stop if person is closer than this

        while self.following:
            try:
                detections = await self.vision.get_detections_from_camera(self.camera_name)
                person = None
                for d in detections:
                    if d.class_name == "Person" and d.confidence > 0.5:
                        person = d
                        break

                if person is None:
                    # No person found, stop and wait
                    await self.base.stop()
                    await asyncio.sleep(0.3)
                    continue

                # Find center of person bounding box
                person_center_x = (person.x_min + person.x_max) / 2
                offset = person_center_x - (FRAME_WIDTH / 2)

                # Check obstacle distance before moving forward
                dist = await self.get_nearest_distance_mm()
                too_close = dist is not None and dist < FOLLOW_DISTANCE_MM

                if abs(offset) < CENTER_ZONE:
                    # Person is centered
                    if too_close:
                        await self.base.stop()
                    else:
                        await self.base.move_straight(distance=100, velocity=150)
                elif offset < -CENTER_ZONE:
                    # Person is to the left
                    await self.base.spin(angle=10, velocity=40)
                else:
                    # Person is to the right
                    await self.base.spin(angle=-10, velocity=40)

                await asyncio.sleep(0.15)

            except Exception as e:
                print(f"Follow loop error: {e}")
                await asyncio.sleep(0.5)

        await self.base.stop()
        print("Follow mode stopped.")

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
            movement_commands = [
                "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT",
                "STOP", "DANCE", "FOLLOW", "STOP_FOLLOWING", "GET_OUT_OF_WAY"
            ]
            if assistant_message.strip().upper() in movement_commands:
                cmd = assistant_message.strip().upper()
                if cmd == "FOLLOW":
                    self.following = True
                    asyncio.create_task(self.follow_loop())
                    return "OK, I'll follow you."
                elif cmd == "STOP_FOLLOWING":
                    self.following = False
                    return "OK, I'll stop following."
                result = await self.execute_movement(assistant_message)
                if result == "obstacle":
                    return "I can't move forward, there's an obstacle less than 1 foot away."
                if cmd == "DANCE":
                    return "That was a good do-si-do!"
                if cmd == "GET_OUT_OF_WAY":
                    return "Sure, moving out of the way!"
                return "OK, moving now."

            # Check if user is asking about distance
            distance_keywords = ["how far", "how close", "distance", "proximity", "near", "wall"]
            if any(kw in user_text.lower() for kw in distance_keywords):
                dist = await self.get_nearest_distance_mm()
                if dist:
                    dist_inches = dist / 25.4
                    assistant_message = f"The nearest object is about {dist_inches:.1f} inches away."
                else:
                    assistant_message = "I couldn't get a distance reading right now."

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
        assistant = OpenAIVoiceAssistant(robot, "wake-word", "speaker", "camera", "base", "Lidar", "vision-1", ser)
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
