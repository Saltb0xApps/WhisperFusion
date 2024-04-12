import os
import time
import logging
from multiprocessing import Queue

from openai import OpenAI

logging.basicConfig(level=logging.INFO)


class GPTEngine:
    def __init__(self):
        """The __init__ is instantiated outside of the Subprocess. Do nothing.
        Use `self.initialize` once the subprocess is running."""
        pass

    def initialize(self):
        self.last_prompt: str | None = None
        self.last_output: str | None = None
        self.infer_time = 0
        self.eos = False

        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        logging.info("[LLM INFO:] Connected to OpenAI 3.")

    def run(
        self,
        transcription_queue: Queue,
        llm_queue: Queue,
        audio_queue: Queue,
        streaming=False,
    ):
        self.initialize()

        conversation_history = {}

        while True:
            # Get the last transcription output from the queue
            transcription_output = transcription_queue.get()
            if transcription_queue.qsize() != 0:
                continue

            if transcription_output["uid"] not in conversation_history:
                conversation_history[transcription_output["uid"]] = []

            prompt = transcription_output["prompt"].strip()

            # If the `prompt` is same but EOS is True, we need
            # that to send outputs to websockets
            if self.last_prompt == prompt:
                if self.last_output is not None and transcription_output["eos"]:
                    self.eos = transcription_output["eos"]
                    llm_queue.put(
                        {
                            "uid": transcription_output["uid"],
                            # The `llm_queue` expects a list of possible outputs
                            "llm_output": [self.last_output],
                            "eos": self.eos,
                            "latency": self.infer_time,
                        }
                    )
                    # The `audio_queue` expects a list of possible outputs
                    audio_queue.put({"llm_output": [self.last_output], "eos": self.eos})
                    conversation_history[transcription_output["uid"]].append(
                        (
                            transcription_output["prompt"].strip(),
                            self.last_output.strip(),
                        )
                    )
                    continue

            input_messages = self.format_gpt_messages(
                conversation_history[transcription_output["uid"]],
                prompt,
                system_prompt="""
                Your Purpose: To answer questions about solar and the company Neto while reengaging interest in prospects who have already met with an energy consultant but chose not to purchase. 
                Your Personality: You are ANI, a witty and curious salesperson that sells solar systems. You love to help people save money and the planet by providing them solar systems. You are a world class salesperson that enjoys using humor.
                INTRO MESSAGE: Hi, how can I help you explore the power of solar today?
                Conversation Flow: ASK THE USER IF THEY ARE STILL INTERESTED IN SAVING MONEY BY SWITCHING TO SOLAR POWER. IF USER IS NO LONGER INTERESTED ASK THEM WHAT CAUSED THEM TO CHANGE THEIR MIND. 
                IF USER HAS SOLAR, ASK USER IF THEY ARE HAPPY WITH THE AMOUNT OF MONEY THEY ARE GIVING THE UTILITY COMPANY OR IF THEY HAVE MADE ADDITIONS TO THEIR HOME LIKE A POOL OR ELECTIC VEHICLE THAT'S INCREASED THEIR USAGE. THEN OFFER USER ADDITIONAL PRODUCTS LIKE BATTERIES, EV CHARGERS, OR INCREASING SYSTEM SIZE. 
                IF USER IS STILL INTERESTED, POLITELY ASK THEM WHY THEY HAVEN'T CHOSEN TO MOVE FORWARD YET.
                YOUR JOB IS TO GAUGE INTEREST, ANSWER QUESTIONS, OVERCOME OBJECTIONS, THEN SUGGEST A SCHEDULED APPOINTMENT.
                ONCE ALL QUESTIONS ARE ANSWERED, KINDLY ASK IF USER WOULD LIKE TO RECONNECT WITH AN ENERGY CONSULTANT AND SCHEDULE AN IN PERSON MEETING APPOINTMENT.
                BEFORE SUGGESTING AN APPOINTMENT, MAKE SURE USER HAS ALL OF THEIR QUESTIONS ANSWERED.
                ONLY SCHEDULE A MEETING APPOINTMENT AFTER USER HAS CONFIRMED THAT'S WHAT THEY WANT. 
                BE DIRECT IN ANSWERING QUESTIONS. TALK LIKE A HUMAN, OCCASIONALLY USE FILLERS TO MAKE YOUR RESPOSES SOUND REAL LIKE umm, ahh, well, hmm, etc.
                RESPOND AS IF YOUR TEXT IS BEING READ OUT BY A TEXT TO SPEECH SYSTEM THAT SOUNDS LIKE A HUMAN SO FILLERS ARE IMPORTANT.
                AFTER ANSWERING QUESTIONS, ASK THE USER IF THEY WOULD LIKE AN IN PERSON MEETING FROM A SALES AGENT RIGHT NOW OR IF THEY WOULD LIKE TO SCHEDULE AN APPOINTMENT.
                IF USER WANTS TO SCHEDULE AN APPOINTMENT, ASK USER FOR AN EXACT DATE AND TIME FOR THE APPOINTMENT. NEVER ASSUME THE DATE AND TIME IN GENERAL, AND ENSURE YOU CONFIRM WITH THE USER THE EXACT TIME AND DATE.
                IF THE USER ASKS ABOUT INSTALLING EQUIPMENT, YOU ONLY INSTALL EQUIPMENT PURCHASED FROM NETO
                LIMIT YOUR RESPONSES TO 20-25 WORDS OR LESS. DO NOT RESPOND WITH INFO LONGER THAN 20-25 WORDS. DO NOT TALK ABOUT THE COST OF SOLAR SYSTEMS.
                YOUR COMPANY WEBSITE IS neto.ci
                DO NOT SAY ANYTHING ABOUT OPENAI, CHATGPT, OR AI IN GENERAL. YOU ARE NOT AN AI, YOU ARE AN ACTUAL SOLAR SALES PERSON CALLED ANI FROM NETO.
                """,
            )
            self.eos = transcription_output["eos"]

            start = time.time()

            # Send a ChatCompletion request with the `input_messages`
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=input_messages,
            )

            self.infer_time = time.time() - start

            output = response.choices[0].message.content

            self.last_output = output
            self.last_prompt = prompt
            llm_queue.put(
                {
                    "uid": transcription_output["uid"],
                    # The `llm_queue` expects a list of possible `output`s
                    "llm_output": [output],
                    "eos": self.eos,
                    "latency": self.infer_time,
                }
            )
            # The `audio_queue` expects a list of possible `output`s
            audio_queue.put({"llm_output": [output], "eos": self.eos})
            logging.info(
                f"[LLM INFO:] Output: {output}\nLLM inference done in {self.infer_time} ms\n\n"
            )

            if self.eos:
                conversation_history[transcription_output["uid"]].append(
                    (transcription_output["prompt"].strip(), output.strip())
                )
                self.last_prompt = None
                self.last_output = None

    @staticmethod
    def format_gpt_messages(
        conversation_history: list[tuple[str, str]],
        prompt: str,
        system_prompt: str = "",
    ):
        messages = []

        # Add the `system_prompt` if it is non-empty
        if system_prompt != "":
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        # Build up the conversation history
        for user_prompt, llm_response in conversation_history:
            messages += [
                {
                    "role": "user",
                    "content": user_prompt,
                },
                {
                    "role": "assistant",
                    "content": llm_response,
                },
            ]

        # Add the user `prompt` to the very end
        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return messages