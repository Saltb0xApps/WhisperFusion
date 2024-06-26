import multiprocessing
import argparse
import threading
import ssl
import time
import sys
import functools
import ctypes
import os

from multiprocessing import Process, Manager, Value, Queue

from whisper_live.trt_server import TranscriptionServer
from gpt_service import GPTEngine
from tts_eleven_service import ElevenLabsTTS


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--whisper_tensorrt_path',
                        type=str,
                        default="/root/TensorRT-LLM/examples/whisper/whisper_small_en",
                        help='Whisper TensorRT model path')
    parser.add_argument('--gpt',
                        action="store_true",
                        help='GPT')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if not args.whisper_tensorrt_path:
        raise ValueError("Please provide whisper_tensorrt_path to run the pipeline.")

    multiprocessing.set_start_method('spawn')
    
    lock = multiprocessing.Lock()
    
    manager = Manager()
    shared_output = manager.list()
    should_send_server_ready = Value(ctypes.c_bool, False)
    transcription_queue = Queue()
    llm_queue = Queue()
    audio_queue = Queue()


    whisper_server = TranscriptionServer()
    whisper_process = multiprocessing.Process(
        target=whisper_server.run,
        args=(
            "0.0.0.0",
            6006,
            transcription_queue,
            llm_queue,
            args.whisper_tensorrt_path,
            should_send_server_ready
        )
    )
    whisper_process.start()

    llm_provider = GPTEngine()
    llm_process = multiprocessing.Process(
        target=llm_provider.run,
        args=(
            transcription_queue,
            llm_queue,
            audio_queue,
        )
    )
    llm_process.start()

    # audio process
    tts_runner = ElevenLabsTTS()
    tts_process = multiprocessing.Process(target=tts_runner.run, args=("0.0.0.0", 8888, os.environ.get("ELEVENLABS_API_KEY"), os.environ.get("ELEVENLABS_VOICE_ID", "pqHfZKP75CvOlQylNhV4"), audio_queue, should_send_server_ready))
    tts_process.start()

    llm_process.join()
    whisper_process.join()
    tts_process.join()
