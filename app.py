import os\
import random
import re
import time
import torch
import soundfile as sf
import librosa
import numpy as np
from PIL import Image
import cv2
import sounddevice as sd
import simpleaudio as sa
import keyboard  # For detecting key presses
from kokoro import KPipeline
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    MoonshineForConditionalGeneration,
    Wav2Vec2Processor,
)
from transformers.image_utils import load_image
from torch.cuda.amp import autocast  # For mixed precision on CUDA (will be a no-op on CPU)
import torch.quantization  # We use PyTorch's built-in dynamic quantization


class VoiceToVoicePipeline:
    def __init__(
        self,
        lang_code='a',
        voice='af_bella',
        images_folder='images',  # Not used when capturing from camera
        vision_model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        moonshine_model_name="UsefulSensors/moonshine-tiny",
        device=None,
        voice_gen=True,
        max_audio_files=None,
        max_new_tokens=100
    ):
        """
        Initialize the VoiceToVoicePipeline.
        """
        self.lang_code = lang_code
        self.voice = voice
        self.images_folder = images_folder
        self.voice_gen = voice_gen
        self.max_audio_files = max_audio_files
        self.max_new_tokens = max_new_tokens

        # Initialize KPipeline for Text-to-Speech
        start_time = time.perf_counter()
        self.pipeline = KPipeline(lang_code=self.lang_code)
        self.pipeline_init_time = time.perf_counter() - start_time

        # Detect device (CPU or CUDA). We'll assume CPU usage if CUDA isn't available.
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize Vision Model
        start_time = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(vision_model_name)

        self.vision_model = AutoModelForVision2Seq.from_pretrained(
            vision_model_name
        ).to(self.device)
        self.vision_model.eval()
        # torch.backends.quantized.engine = 'qnnpack'
        # Dynamic quantization for CPU inference (works on nn.Linear layers)
        if self.device == "cpu":
            if torch.backends.quantized.engine == 'none':
                print("No quantization engine found. Setting quantized engine to 'qnnpack'")
                torch.backends.quantized.engine = 'none'
            try:
                self.vision_model = torch.quantization.quantize_dynamic(
                    self.vision_model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            except Exception as e:
                print("Dynamic quantization failed, continuing without quantization. Error:", e)

        self.vision_model_init_time = time.perf_counter() - start_time

        # Initialize Moonshine for Speech-to-Text (unquantized)
        start_time = time.perf_counter()
        self.moonshine_model = MoonshineForConditionalGeneration.from_pretrained(moonshine_model_name).to(self.device)
        self.moonshine_model.eval()  # Set to evaluation mode
        self.moonshine_processor = Wav2Vec2Processor.from_pretrained(moonshine_model_name)
        self.moonshine_init_time = time.perf_counter() - start_time

        # Define vision-related patterns
        self.vision_patterns = [
            r"\bwhat am i looking at\b",
            r"\bwhat does this image do\b",
            r"\btell me about this picture\b",
            r"\bdescribe this image\b",
            r"\bwhat is in this image\b"
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.vision_patterns]

        # Dictionary to store timing information
        self.timing_info = {}

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio to text using Moonshine.
        """
        start_time = time.perf_counter()
        audio_array, sampling_rate = sf.read(audio_path)

        # Convert to mono if necessary
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample to 16000 Hz if necessary using librosa
        if sampling_rate != 16000:
            resample_start = time.perf_counter()
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000, res_type='kaiser_fast')
            sampling_rate = 16000
            self.timing_info['resample_time'] = time.perf_counter() - resample_start
        else:
            self.timing_info['resample_time'] = 0.0

        prepare_start = time.perf_counter()
        with torch.no_grad():
            inputs = self.moonshine_processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).input_values.to(self.device)
        self.timing_info['prepare_transcription_time'] = time.perf_counter() - prepare_start

        generate_start = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.moonshine_model.generate(
                inputs,
                max_length=self.max_new_tokens,
                num_beams=1,
                do_sample=False
            )
            transcription = self.moonshine_processor.decode(generated_ids[0], skip_special_tokens=True)
        self.timing_info['transcription_time'] = time.perf_counter() - generate_start

        total_time = time.perf_counter() - start_time
        self.timing_info['total_transcription_time'] = total_time

        print(f"Transcription: {transcription}")
        return transcription

    def is_vision_query(self, text):
        """
        Check if the transcribed text matches any vision-related patterns.
        """
        start_time = time.perf_counter()
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                self.timing_info['match_time'] = time.perf_counter() - start_time
                print(f"Matched pattern: {pattern.pattern}")
                return True
        self.timing_info['match_time'] = time.perf_counter() - start_time
        return False

    def capture_camera_image(self):
        """
        Capture an image from the system camera and save it as a temporary file.
        """
        print("Capturing image from camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open the camera.")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise Exception("Failed to capture image from camera.")
        # Convert BGR (OpenCV) to RGB (PIL)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        temp_image_path = "temp_camera.jpg"
        image.save(temp_image_path)
        print(f"Captured image saved to {temp_image_path}")
        return temp_image_path

    def process_image(self, image_path):
        """
        Process the image using the vision model to generate a textual description.
        """
        start_time = time.perf_counter()
        image = load_image(image_path)

        # Resize image to reduce compute
        resize_start = time.perf_counter()
        image = image.resize((128, 128))
        self.timing_info['resize_time'] = time.perf_counter() - resize_start

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe the content of this image."}
                ]
            },
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        prepare_start = time.perf_counter()
        with torch.no_grad():
            inputs = self.processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            ).to(self.device)
        self.timing_info['prepare_image_time'] = time.perf_counter() - prepare_start

        generate_start = time.perf_counter()
        if self.device == "cuda":
            with autocast():
                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False
                )
        else:
            with torch.no_grad():
                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False
                )
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.timing_info['image_generation_time'] = time.perf_counter() - generate_start

        description = generated_texts[0].strip()
        self.timing_info['process_image_time'] = time.perf_counter() - start_time

        print(f"Generated description: {description}")
        return description

    def process_text(self, text):
        """
        Process general text query using the vision model to generate a response.
        """
        text = text[:512]

        start_time = time.perf_counter()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                ]
            },
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        prepare_start = time.perf_counter()
        with torch.no_grad():
            inputs = self.processor(
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
        self.timing_info['prepare_text_time'] = time.perf_counter() - prepare_start

        generate_start = time.perf_counter()
        if self.device == "cuda":
            with autocast():
                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False
                )
        else:
            with torch.no_grad():
                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False
                )
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.timing_info['text_generation_time'] = time.perf_counter() - generate_start

        response = generated_texts[0].strip()
        self.timing_info['process_text_time'] = time.perf_counter() - start_time

        print(f"Generated response: {response}")
        return response

    def generate_speech(self, text, output_path):
        """
        Generate speech from text using KPipeline, save as WAV files, and play them.
        """
        if not self.voice_gen:
            print("Voice generation is disabled. Skipping speech generation.")
            return

        start_time = time.perf_counter()
        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=1,
            split_pattern=r'\n+'
        )
        generate_start = time.perf_counter()
        audio_files_generated = 0

        for i, (gs, ps, audio) in enumerate(generator):
            if self.max_audio_files is not None and audio_files_generated >= self.max_audio_files:
                print(f"Reached maximum of {self.max_audio_files} audio files. Stopping speech generation.")
                break
            print(f"Processing segment {i}: {gs}")
            wav_filename = f"{output_path}_{i}.wav"
            sf.write(wav_filename, audio, 24000)
            # Play the generated audio segment
            print(f"Playing generated audio: {wav_filename}")
            wave_obj = sa.WaveObject.from_wave_file(wav_filename)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            audio_files_generated += 1

        self.timing_info['speech_generation_time'] = time.perf_counter() - generate_start
        self.timing_info['speech_total_time'] = time.perf_counter() - start_time

        if self.voice_gen:
            print("Speech generated, saved, and played.")

    def handle_audio_input(self, audio_path, output_audio_path="output"):
        """
        Handle the entire pipeline from audio input to audio output.
        """
        overall_start = time.perf_counter()

        # 1. Transcribe the user audio
        t_start = time.perf_counter()
        transcription = self.transcribe_audio(audio_path)
        self.timing_info['handle_transcription_time'] = time.perf_counter() - t_start

        # 2. Check if it's a vision-related query
        v_start = time.perf_counter()
        is_vision = self.is_vision_query(transcription)
        self.timing_info['handle_vision_query_time'] = time.perf_counter() - v_start

        if is_vision:
            # 3. Capture image from system camera
            ci_start = time.perf_counter()
            image_path = self.capture_camera_image()
            self.timing_info['handle_camera_capture_time'] = time.perf_counter() - ci_start

            # 4. Process the captured image
            pi_start = time.perf_counter()
            description = self.process_image(image_path)
            self.timing_info['handle_process_image_time'] = time.perf_counter() - pi_start

            # 5. Generate and play speech from the description
            gs_start = time.perf_counter()
            self.generate_speech(description, output_audio_path)
            self.timing_info['handle_generate_speech_time'] = time.perf_counter() - gs_start
        else:
            # Process as general text
            pt_start = time.perf_counter()
            response = self.process_text(transcription)
            self.timing_info['handle_process_text_time'] = time.perf_counter() - pt_start

            # Generate and play speech from the response
            self.generate_speech(response, output_audio_path)

        self.timing_info['overall_time'] = time.perf_counter() - overall_start

    def print_timing_info(self):
        """
        Print the collected timing information.
        """
        print("\n--- Timing Information ---")
        for key, value in self.timing_info.items():
            print(f"{key}: {value:.4f} seconds")
        print("--------------------------\n")


import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from pynput import keyboard  # Use pynput instead of keyboard

def record_audio_while_r(sample_rate=16000, output_filename="temp_input.wav"):
    """
    Record audio continuously while the "r" key is held down.
    Once "r" is released, stop recording and save the audio.
    """
    print("Press and hold the 'r' key to start recording. Release 'r' to stop.")

    recording = []
    recording_flag = {"recording": False}  # Use dict to allow modification in inner functions

    def on_press(key):
        try:
            if key.char == 'r' and not recording_flag["recording"]:
                recording_flag["recording"] = True
                print("Recording started...")
        except AttributeError:
            pass

    def on_release(key):
        try:
            if key.char == 'r':
                recording_flag["recording"] = False
                print("Recording stopped.")
                # Stop the listener when "r" is released
                return False
        except AttributeError:
            pass

    # Start listening to key events
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Wait until the recording starts
    while not recording_flag["recording"]:
        time.sleep(0.01)

    # Record audio until recording_flag["recording"] becomes False
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
        while recording_flag["recording"]:
            data, _ = stream.read(int(sample_rate * 0.1))  # Read in chunks of 0.1 seconds
            recording.append(data.copy())

    # Ensure the listener has stopped
    listener.join()

    # Concatenate all recorded chunks and save to file
    recording = np.concatenate(recording, axis=0)
    sf.write(output_filename, recording, sample_rate)
    print(f"Recording saved to {output_filename}")
    return output_filename


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voice to Voice Pipeline with Mic and Camera Input (Record while 'r' is pressed)")
    parser.add_argument("--output_audio", type=str, default="output", help="Base path for the output audio file.")
    parser.add_argument("--voice_gen", action='store_true', help="Enable speech generation.")
    parser.add_argument("--no_voice_gen", action='store_false', dest='voice_gen', help="Disable speech generation.")
    parser.set_defaults(voice_gen=True)
    parser.add_argument("--max_audio_files", type=int, default=None,
                        help="Maximum number of audio files to generate per response.")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="Maximum number of tokens to generate for responses.")
    args = parser.parse_args()

    def main():
        # Record audio from system mic while "r" is held down
        input_audio_path = record_audio_while_r()
        
        pipeline = VoiceToVoicePipeline(
            lang_code='a',
            voice='af_bella',
            images_folder="",  # Not used when capturing from camera
            vision_model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
            moonshine_model_name="UsefulSensors/moonshine-tiny",
            device=None,  # Auto-detect CPU or GPU
            voice_gen=args.voice_gen,
            max_audio_files=args.max_audio_files,
            max_new_tokens=args.max_new_tokens
        )
        pipeline.handle_audio_input(input_audio_path, args.output_audio)
        pipeline.print_timing_info()

    main()
