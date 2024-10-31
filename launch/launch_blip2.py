import sys
import argparse

from io import BytesIO
import json
import base64

from PIL import Image
import numpy as np

import torch

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def load_model():
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    blip_v2_model_type = "blip2-flan-t5-xl"
        
    with torch.cuda.device("cuda"):
        processor = Blip2Processor.from_pretrained(f"Salesforce/{blip_v2_model_type}")
        model = Blip2ForConditionalGeneration.from_pretrained(
            f"Salesforce/{blip_v2_model_type}",
            torch_dtype="auto",
            device_map="sequential"
        )
    # qa_prompt = "Question: {} Short answer:"
    # caption_prompt = "a photo of"
    model = model.eval()
    return processor, model


def predict(image, question):
    inputs = processor(images=image, text=question, return_tensors="pt", padding="longest").to("cuda")
    generated_ids = model.generate(**inputs, length_penalty=-1, num_beams=5, max_length=500, min_length=1,
                                   do_sample=True, top_p=0.9, repetition_penalty=1.0,
                                   num_return_sequences=1, temperature=1)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

processor, model = load_model()
    
app = FastAPI() 

@app.post("/generate")
async def process_payload(request: dict):
    question = request["text"]
    encoded_image = request["image_data"]
    
    image_bytes = BytesIO(base64.b64decode(encoded_image))
    image = Image.open(image_bytes)
    
    response = predict(image, question)
    return JSONResponse(content={"text": response})
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7000)
    args = parser.parse_args()
    
    port = args.port
    
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port)
