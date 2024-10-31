"""
Modules for interfacing with models in TraveLER.
"""

import abc
import asyncio
import base64
import io
import multiprocessing as mp
import os
import pickle
from io import BytesIO
from typing import List, Union

import aiohttp
import numpy as np
import openai
import requests
import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

try:
    config_path = os.path.join(os.environ["EXP_PATH"], "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
except Exception:
    print("NO CONFIG FOUND")
    config = {}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================== Base abstract model ========================== #
class BaseModel(abc.ABC):

    def __init__(self, port_number=None):
        self.port_number = port_number

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

# ========================== Specific Models ========================== #


class GPT(BaseModel):
    
    def __init__(self, max_tries=1):
        super().__init__()
        
        self.temperature = config["llm"]["temperature"]
        self.model = config["llm"]["model"]
        self.max_tries = max_tries

    @staticmethod
    def generate(
        prompt,
        model,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=4096,
        n=1,
        temperature=1,
        max_tries=3,
    ):
        for _ in range(max_tries):
            try:
                completion = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                    ],
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens,
                    n=n,
                    temperature=temperature,
                )
                output_message = completion.choices[0].message.content
                return output_message
            except Exception as e:
                print("===")
                print("ERROR:", e)
                print("=== TRYING AGAIN ===")
                continue
        print("Unable to get response from LLM")
        return None

    @staticmethod
    def batch_generate(x):
        return GPT.generate(*x)

    def forward(self, prompts: Union[List[str], str], process_name=None):
        
        # batched
        if isinstance(prompts, list):
            with mp.Pool(processes=mp.cpu_count()) as pool:
                response = pool.map(
                    self.batch_generate, [(prompt, self.model) for prompt in prompts]
                )
        else:
        # single
            response = GPT.generate(prompts, self.model)
        return response


class GPT_4V(BaseModel):
    
    def __init__(self):
        super().__init__()

    @staticmethod
    def encode_image(image_file):
        image_bytes = io.BytesIO()
        image_file.save(image_bytes, format="PNG")
        image_byte_array = image_bytes.getvalue()
        return base64.b64encode(image_byte_array).decode("utf-8")

    def construct_payload(self, image, prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}",
        }
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 200,
        }
        return headers, payload

    def forward(self, image_list, questions):
        outputs = []
        for image, question in zip(image_list, questions):
            try:
                base64_image = self.encode_image(image)
                headers, payload = self.construct_payload(base64_image, question)
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                output = response.json()["choices"][0]["message"]["content"]
                outputs.append(output)
            except Exception as e:
                print("ERROR:", e)
                return None
        return outputs


class LLaVA_13B(BaseModel):

    def __init__(self, port_number=8000):
        super().__init__()
        self.url = f"http://localhost:{port_number}/generate"

    @staticmethod
    def encode_image_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    async def send_request(url, data, delay=0):
        await asyncio.sleep(delay)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as resp:
                output = await resp.json()
        return output

    @staticmethod
    async def run(url, image_list, questions):
        response = []
        for image, question in zip(image_list, questions):
            payload = (
                url,
                {
                    "text": f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:",
                    "image_data": LLaVA_13B.encode_image_base64(image),
                    "sampling_params": {
                        "max_new_tokens": config["vlm"]["max_new_tokens"],
                    },
                },
            )
            response.append(LLaVA_13B.send_request(*payload))

        rets = await asyncio.gather(*response)
        outputs = []
        for ret in rets:
            outputs.append(ret["text"])
        response = None
        return outputs

    def forward(self, image_list, questions):
        assert len(image_list) == len(questions)
        return asyncio.run(self.run(self.url, image_list, questions))


class LLaVA_34B(BaseModel):

    def __init__(self, port_number=8000):
        super().__init__()
        self.url = f"http://localhost:{port_number}/generate"
        self.tokenizer = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.6-34b")

    @staticmethod
    def encode_image_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    async def send_request(url, data, delay=0):
        await asyncio.sleep(delay)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as resp:
                output = await resp.json()
        return output

    @staticmethod
    async def run(url, image_list, chat_strs):
        response = []
        for image, chat_str in zip(image_list, chat_strs):
            payload = (
                url,
                {
                    "text": chat_str,
                    "image_data": LLaVA_34B.encode_image_base64(image),
                    "sampling_params": {
                        "max_new_tokens": config["llava"]["max_new_tokens"],
                    },
                },
            )
            response.append(LLaVA_34B.send_request(*payload))

        rets = await asyncio.gather(*response)
        outputs = []
        for ret in rets:
            outputs.append(ret["text"])
        response = None
        return outputs

    def forward(self, image_list, questions):
        assert len(image_list) == len(questions)
        chat_strs = []
        for question in questions:
            chat = [
                {"role": "system", "content": "Answer the question."},
                {"role": "user", "content": "<image>\n" + question},
            ]

            chat_str = self.tokenizer.apply_chat_template(chat, tokenize=False)
            chat_str += "<|img_start|>assistant\n"
            chat_strs.append(chat_str)
        return asyncio.run(self.run(self.url, image_list, chat_strs))


class BLIP_2(BaseModel):

    def __init__(self, port_number=8000):
        super().__init__()
        self.url = f"http://localhost:{port_number}/generate"

    @staticmethod
    def encode_image_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    async def send_request(url, data, delay=0):
        await asyncio.sleep(delay)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as resp:
                output = await resp.json()
        return output

    @staticmethod
    async def run(url, image_list, questions):
        response = []
        for image, question in zip(image_list, questions):
            payload = (
                url,
                {
                    "text": f"{question}",
                    "image_data": BLIP_2.encode_image_base64(image),
                    "sampling_params": {
                        "max_new_tokens": config["llava"]["max_new_tokens"],
                    },
                },
            )
            response.append(BLIP_2.send_request(*payload))

        rets = await asyncio.gather(*response)
        outputs = []
        for ret in rets:
            outputs.append(ret["text"])
        response = None
        return outputs

    def forward(self, image_list, questions):
        assert len(image_list) == len(questions)
        return asyncio.run(self.run(self.url, image_list, questions))


class LaViLa(BaseModel):

    def __init__(self, port_number=8000):
        super().__init__()
        self.url = f"http://localhost:{port_number}/generate"

    @staticmethod
    async def send_request(url, data, delay=0):
        await asyncio.sleep(delay)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as resp:
                output = await resp.json()
        return output

    @staticmethod
    async def run(url, frame_ids, path):
        response = []
        payload = (
            url,
            {
                "video_name": path,
                "frame_ids": frame_ids,
            },
        )
        response.append(LaViLa.send_request(*payload))

        rets = await asyncio.gather(*response)
        outputs = []
        for ret in rets:
            outputs.append(ret["caption"])
        response = None
        return outputs

    def forward(self, frame_ids, path):
        return asyncio.run(self.run(self.url, frame_ids, path))


class Llama_3(BaseModel):

    def __init__(self, port_number=8000, max_tries=1):
        super().__init__()
        
        self.temperature = config["gpt"]["temperature"]
        # self.model = "NousResearch/Meta-Llama-3-8B-Instruct"
        self.model = "NousResearch/Meta-Llama-3-70B-Instruct"
        self.max_tries = max_tries
        self.port_number = port_number
        self.client = OpenAI(
            base_url=f"http://localhost:{port_number}/v1",
            api_key="token-abc123",
        )

    @staticmethod
    def generate(
        prompt,
        model,
        port_number,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=1000,
        n=1,
        temperature=1,
        max_tries=3,
    ):
        for _ in range(max_tries):
            try:
                client = OpenAI(
                    base_url=f"http://localhost:{port_number}/v1",
                    api_key="token-abc123",
                )
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                    ],
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens,
                    n=n,
                    temperature=temperature,
                )
                output_message = completion.choices[0].message.content
                return output_message
            except Exception as e:
                print("===")
                print("ERROR:", e)
                print("=== TRYING AGAIN ===")
                continue
        print("Unable to get response from LLM")
        return None

    @staticmethod
    def batch_generate(x):
        return Llama_3.generate(*x)

    def forward(self, prompts: Union[List[str], str]):
        
        # batched
        if isinstance(prompts, list):
            with mp.Pool(processes=mp.cpu_count()) as pool:
                response = pool.map(
                    self.batch_generate,
                    [(prompt, self.model, self.port_number) for prompt in prompts],
                )
        else:
        # single
            response = Llama_3.generate(prompts, self.model, self.port_number)
        return response
