import functools
import time
import logging
import requests

from tqdm import tqdm
from websockets.sync.server import serve

logging.basicConfig(level=logging.INFO)

class ElevenLabsTTS:
    def __init__(self):
        pass

    def initialize_model(self, api_key, voice_id):
        self.api_key = api_key
        self.voice_id = voice_id
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        self.endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
    
        # Test the API connection with a warm-up request
        logging.info("\n[ElevenLabs INFO:] Warming up ElevenLabs TTS API. Please wait ...\n")
        data = {
            "text": "Hello, I am warming up.",
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        response = requests.post(self.endpoint, json=data, headers=self.headers)
        if response.status_code == 200:
            logging.info("[ElevenLabs INFO:] API warmup successful.")
        else:
            logging.warning(f"[ElevenLabs WARNING:] API warmup failed with status code {response.status_code}")
        logging.info("[ElevenLabs INFO:] Warmed up ElevenLabs TTS API. Connect to the WebGUI now.")
        self.last_llm_response = None
        self.last_api_request = None

    def run(self, host, port, api_key, voice_id, audio_queue=None, should_send_server_ready=None):
        self.initialize_model(api_key=api_key, voice_id=voice_id)
        should_send_server_ready.value = True

        with serve(
            functools.partial(self.start_elevenlabs_tts, audio_queue=audio_queue),
            host, port
            ) as server:
            server.serve_forever()

    def start_elevenlabs_tts(self, websocket, audio_queue=None):
        self.eos = False
        self.output_audio = None

        while True:
            llm_response = audio_queue.get()
            if audio_queue.qsize() != 0:
                continue

            try:
                websocket.ping()
            except Exception as e:
                del websocket
                audio_queue.put(llm_response)
                break

            llm_output = llm_response["llm_output"][0]
            self.eos = llm_response["eos"]

            if self.last_llm_response != llm_output.strip():
                self.last_llm_response = llm_output.strip()
                try:
                    start = time.time()
                    if self.last_api_request is not None and self.last_api_request == llm_output.strip():
                        logging.info("[ElevenLabs INFO:] Skipping duplicate request.")
                        continue

                    self.last_api_request = llm_output.strip()
                    response = requests.post(self.endpoint, json={
                        "text": llm_output.strip(),
                        "model_id": "eleven_turbo_v2",
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.5
                        }
                    }, headers=self.headers)
                    self.output_audio = response.content 
                    
                    inference_time = time.time() - start
                    logging.info(f"[ElevenLabs INFO:] TTS inference done in {inference_time:.2f} seconds.")
                except Exception as e:
                    logging.error(f"[ElevenLabs ERROR:] Error during TTS request: {e}")
                    continue

            if self.eos and self.output_audio is not None:
                try:
                    websocket.send(self.output_audio)
                except Exception as e:
                    logging.error(f"[WhisperSpeech ERROR:] Audio error: {e}")