"""
Main inference script to run experiments for TraveLER.
"""

import argparse
import importlib
import os
import time
import traceback

import yaml
from tqdm import tqdm

import wandb
from dataset import (
    CausalVidQADataset,
    EgoSchemaDataset,
    NextQADataset,
    PerceptionTestDataset,
    STARDataset,
)


def main(args):
    
    # Set the experiment path as an environment variable. Used for config file, results, prompts.
    os.environ['EXP_PATH'] = os.path.join(os.getcwd(), "experiments", args.exp)
    os.environ["OUTFILE_NAME"] = args.outfile_name
    
    config_path = os.path.join(os.environ["EXP_PATH"], "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup wandb
    if config["wandb"]:
        wandb.init(
        # Set the project where this run will be logged
        project="traveler", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{args.exp + '_' + args.outfile_name}", 
        # Track hyperparameters and run metadata
        config=config
    )
    
    # Setup results dir
    if not os.path.exists(os.path.join(os.getcwd(), config["results_dir"], config["experiment_name"])):
        os.makedirs(os.path.join(os.getcwd(), config["results_dir"], config["experiment_name"]))
    
    modules = importlib.import_module("modules")
    answerer = importlib.import_module(f"experiments.{args.exp}.answerer") # Imports the answerer for the experiment
    
    # Load Dataset
    if config["dataset"]["name"] == 'nextqa':
        dataset = NextQADataset(
            data_path=config["dataset"]["data_path"],
            query_file=config["dataset"]["query_file"],
            start_sample=args.start_sample,
            max_samples=args.max_samples
        )
    elif config["dataset"]["name"] == 'perception_test':
        dataset = PerceptionTestDataset(
            data_path=config["dataset"]["data_path"],
            query_file=config["dataset"]["query_file"],
            start_sample=args.start_sample,
            max_samples=args.max_samples
        )
    elif config["dataset"]["name"] == 'star':
        dataset = STARDataset(
            data_path=config["dataset"]["data_path"],
            query_file=config["dataset"]["query_file"],
            start_sample=args.start_sample,
            max_samples=args.max_samples
        )
    elif config["dataset"]["name"] == 'egoschema':
        dataset = EgoSchemaDataset(
            data_path=config["dataset"]["data_path"],
            query_file=config["dataset"]["query_file"],
            start_sample=args.start_sample,
            max_samples=args.max_samples,
            # evaluation=True
        )
    elif config["dataset"]["name"] == 'causalvidqa':
        dataset = CausalVidQADataset(
            data_path=config["dataset"]["data_path"],
            query_file=config["dataset"]["query_file"],
            start_sample=args.start_sample,
            max_samples=args.max_samples
        )
    else:
        raise Exception(f"Dataset <{config['dataset']['name']}> not found.")
        
    # Load Models
    if config["vlm"]["model"] == 'llava-1.6-13b':
        vlm = modules.LLaVA_13B(port_number=args.vlm_port)
    elif config["vlm"]["model"] == 'llava-1.6-34b':
        vlm = modules.LLaVA_34B(port_number=args.vlm_port)
    elif config["vlm"]["model"] == 'lavila':
        vlm = modules.LaViLa(port_number=args.vlm_port)
    elif config["vlm"]["model"] == 'gpt-4v':
        vlm = modules.GPT_4V()
    elif config["vlm"]["model"] == 'blip2':
        vlm = modules.BLIP_2(port_number=args.vlm_port)
    else:
        raise Exception(f"Multimodal model <{config['vlm']['model']}> not found.")
    
    if "gpt" in config["llm"]["model"]:
        llm = modules.GPT()
    elif config["llm"]["model"] == 'llama3':
        llm = modules.Llama_3(port_number=args.llm_port)
    else:
        raise Exception(f"Large language model <{config['llm']['model']}> not found.")
    
    ans = answerer.Answerer(caption_model=vlm, vqa_model=vlm, llm=llm)

    # Creates queue
    print("Creating queue.")
    q = []
    for i in tqdm(range(len(dataset))):
        q.append(i)
    
    # Processes queue and writes to output file
    print("Evaluating.")
    total_correct = 0
    total = 0
    pbar = tqdm(total=len(dataset))
    while q:
        if config["wandb"]:
            start_time = time.time()
        idx = q.pop(0)
        item = dataset[idx]

        video_obj = dataset.construct_video(item)
        try:
            if dataset.reasons:
                pred, reason = ans.forward(video_obj)
            else:
                pred = ans.forward(video_obj)


            with open(os.path.join(os.getcwd(), config["results_dir"], config["experiment_name"], f"output_{args.outfile_name}.tsv"), "a") as outfile:
                # reasons
                if dataset.reasons:
                    outfile.write(f"{item['index'] + args.start_sample}\t{item['video_name']}\t{item['query_type']}\t{item['query']}\t{item['possible_answers']}\t{pred}\t{item['possible_reasons']}\t{reason}\n")
                # no answer
                elif dataset.evaluation:
                    outfile.write(f"{item['index'] + args.start_sample}\t{item['video_name']}\t{item['query_type']}\t{item['query']}\t{item['possible_answers']}\t{pred}\n")
                # start and end
                elif dataset.segment:
                    outfile.write(f"{item['index'] + args.start_sample}\t{item['video_name']}\t{item['query_type']}\t{item['query']}\t{item['answer']}\t{item['possible_answers']}\t{item['start']}\t{item['end']}\t{pred}\n")
                else:
                    outfile.write(f"{item['index'] + args.start_sample}\t{item['video_name']}\t{item['query_type']}\t{item['query']}\t{item['answer']}\t{item['possible_answers']}\t{pred}\n")

            if not dataset.evaluation:
                if pred == item["answer"]:
                    print("correct")
                    total_correct += 1
                if pred != item["answer"]:
                    print("incorrect")
            total += 1
            if dataset.evaluation:
                print(total)

            if config["wandb"]:
                if not dataset.evaluation:
                    wandb.log({"total_accuracy": total_correct / total})
                wandb.log({"total": total})
                end_time = time.time()
                time_taken = end_time - start_time
                wandb.log({"time": time_taken})

        except Exception:
            print(traceback.format_exc())
            q.append(idx)
            continue
        
        video_obj = None
        ans.video = None
        item = None
        
        pbar.update(1)
    
    pbar.close()
    
    if config["wandb"]:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--outfile_name", type=str, default="")
    parser.add_argument("--llm_port", type=int, default=7000)
    parser.add_argument("--vlm_port", type=int, default=8000)
    args = parser.parse_args()

    main(args)
