import argparse
import os
import urllib.request
from collections import OrderedDict

import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import decord

from lavila.data.video_transforms import Permute
from lavila.data.datasets import get_frame_ids, video_loader_by_frames
from lavila.models.models import VCLM_OPENAI_TIMESFORMER_BASE_GPT2
from lavila.models.tokenizer import MyGPT2Tokenizer
from lavila.models.utils import inflate_positional_embeds
from eval_narrator import decode_one

from fastapi import FastAPI
from fastapi.responses import JSONResponse



# ckpt_name = 'vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth'
# ckpt_path = os.path.join('modelzoo/', ckpt_name)
# os.makedirs('modelzoo/', exist_ok=True)
# if not os.path.exists(ckpt_path):
#     print('downloading model to {}'.format(ckpt_path))
#     urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator/{}'.format(ckpt_name), ckpt_path)
ckpt_path = '/home/amosyou/lavila/ckpt_base.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = OrderedDict()
for k, v in ckpt['state_dict'].items():
    state_dict[k.replace('module.', '')] = v

# instantiate the model, and load the pre-trained weights
model = VCLM_OPENAI_TIMESFORMER_BASE_GPT2(
    text_use_cls_token=False,
    project_embed_dim=256,
    gated_xattn=True,
    timesformer_gated_xattn=False,
    freeze_lm_vclm=False,
    freeze_visual_vclm=False,
    freeze_visual_vclm_temporal=False,
    num_frames=4,
    drop_path_rate=0.
)
model.load_state_dict(state_dict, strict=True)
model.cuda()
model.eval()

def predict(video_path, frame_ids):
    # vr = decord.VideoReader(video_path)
    frames = video_loader_by_frames('./', video_path, frame_ids)

    # transforms on input frames
    crop_size = 224
    val_transform = transforms.Compose([
        Permute([3, 0, 1, 2]),
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])
    ])
    
    frames = val_transform(frames)
    frames = frames.unsqueeze(0)  # fake a batch dimension
    frames = frames.to("cuda")
    
    tokenizer = MyGPT2Tokenizer('gpt2', add_bos=True)
    with torch.no_grad():
        image_features = model.encode_image(frames)
        generated_text_ids, ppls = model.generate(
            image_features,
            tokenizer,
            target=None, # free-form generation
            max_text_length=150,
            top_k=None,
            top_p=0.95,  # nucleus sampling
            num_return_sequences=3, # number of candidates: 10
            temperature=0.7,
            early_stopping=True,
        )

    output_string = ""
    for i in range(3):
        generated_text_str = decode_one(generated_text_ids[i], tokenizer)
        generated_text_str = generated_text_str.replace('#C', '').strip()
        generated_text_str = generated_text_str.replace('#O', '').strip()
        output_string += generated_text_str + ". "
    return output_string
    
    

app = FastAPI() 

@app.post("/generate")
async def process_payload(request: dict):
    frame_ids = request["frame_ids"]
    video_name = request["video_name"]
    video_path = "/home/amosyou/egoschema/videos/videos/"
    video_path = os.path.join(video_path, video_name)
    # question = request["question"]
    
    response = predict(video_path, frame_ids)
    return JSONResponse(content={"caption": response})
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7000)
    args = parser.parse_args()
    
    port = args.port
    
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port)