# TraveLER: A Modular Multi-LMM Agent Framework for Video Question-Answering

This repo contains the code for the paper [TraveLER: A Modular Multi-LMM Agent Framework for Video Question-Answering](https://arxiv.org/abs/2404.01476), published at EMNLP 2024. Check out our project page [here](https://traveler-framework.github.io/)!


## Setup

Follow these steps to setup the repo.

1. Setup a Conda environment with the required packages.

```bash
conda create -n traveler python=3.9
conda activate traveler
pip install -r requirements.txt
```

2. Create a `.env` file that contains your `OPENAI_API_KEY`.

3. Login to `wandb` account.

```
wandb login
```

### Datasets

For each dataset, ensure that the videos are moved into the same directory. This will be the `data_path` in the config files.

### Models

We use a LaViLa checkpoint that is trained on the Ego4D videos that don't overlap with EgoSchema, provided by [LLoVi](https://github.com/CeeZh/LLoVi). You can find the model checkpoint [here](https://drive.google.com/file/d/1AZ5I4eTUAUBX31rL8jLB00vNrlFGWiv8/view). In order to serve LaViLa, clone the [repo](https://github.com/facebookresearch/LaViLa) and copy in `launch/launch_lavila.py` to the root directory of the cloned repo.

We serve LLaVA-1.6 using [SGLang](https://github.com/sgl-project/sglang), and Llama 3 using [vLLM](https://github.com/vllm-project/vllm).

## Experiments

### Config

The config files are found under each experiment. These config files determine the dataset being evaluated, the models, various paths, and other hyperparameters.

If you want to evaluate TraveLER on a new dataset, be sure to set the correct dataset info, including the name, path, and query file. You will also need to create your own query file, examples can be found under `data/`.


### Launching Servers

Before we run experiments, we need to launch the servers for the VLM/LLMs.

1. Launch VLM server. The port number can be changed to your preference.

```bash
# LLaVA-1.6
CUDA_VISIBLE_DEVICES=0 bash launch/launch_llava.sh --port 30000

# LaViLa (from root of LaViLa repo)
CUDA_VISIBLE_DEVICES=0 python3 launch_lavila.py --port 30000
```

2. For models served by SGLang (LLaVA-1.6, not LaViLa), we need to launch a wrapper script to forward incoming asynchronous requests to SGLang. The `sglang_port` has to match the port number we set earlier, but the `wrapper_port` can be changed to your preference. We will send requests to the VLM using the `wrapper_port` instead of `sglang_port`.

```
python3 launch/launch_wrapper.py --sglang_port 30000 --wrapper_port 8000
```

3. (Optional) - Launch LLM server for local LLMs.

```bash
# Llama 3
CUDA_VISIBLE_DEVICES=1 vllm serve NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123
```


### Benchmark

In general, this is the command to use for running experiments.

```bash
python3 main.py --exp <experiment_name> --start_sample <start sample> --max_samples <max samples> --outfile_name <batch_name> --vlm_port 8000
```

Note: The `vlm_port` should match the port number we set for the `wrapper_port`, instead of `sglang_port`. If a local LLM is also used, we need to set the port number of the LLM using `llm_port`.

For example, to evaluate NExT-QA:

```bash
# first 100 examples
python3 main.py --exp nextqa --start_sample 0 --max_samples 100 --outfile_name batch_0 --vlm_port 8000

# next 100 examples
python3 main.py --exp nextqa --start_sample 100 --max_samples 100 --outfile_name batch_1 --vlm_port 8000

...
```

We manually define the start sample and max number of samples to allow fine-grain control over how we partition the dataset across GPUs. For our workload on NVIDIA RTX 6000 Ada's, we could have 5 processes sharing the same GPU for the VLM. This number will change depending on what GPU you have due to different memory amounts for the KV cache.

After the output files are generated, we can find the accuracy by running the eval script.

```bash
# python3 eval.py --exp <experiment_name>
python3 eval.py --exp nextqa
```

## Acknowledgements

Code for configs, dataloaders, and query files adapted from [RVP](https://github.com/para-lost/RVP) and [ViperGPT](https://github.com/cvlab-columbia/viper).


## Citation

Our paper can be cited as:

```latex
@misc{shang2024traveler,
    title={TraveLER: A Modular Multi-LMM Agent Framework for Video Question-Answering},
    author={Chuyi Shang and Amos You and Sanjay Subramanian and Trevor Darrell and Roei Herzig},
    year={2024},
    eprint={2404.01476},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2404.01476}, 
}
```
