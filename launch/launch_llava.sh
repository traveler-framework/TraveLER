#!/bin/bash

# Initialize variables for named arguments
model_name="llava-v1.6-vicuna-13b"
# model_name="llava-v1.6-34b"
# model_name="llava-v1.5-13b"
tokenizer_name="llava-hf/llava-1.5-13b-hf"
# tokenizer_name="liuhaotian/llava-v1.6-34b-tokenizer"
port_number="30000"
mem_fraction_static="0.95"  # Default memory fraction

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            model_name="$2"
            shift # past argument
            shift # past value
            ;;
        --tokenizer)
            tokenizer_name="$2"
            shift # past argument
            shift # past value
            ;;
        --port)
            port_number="$2"
            shift # past argument
            shift # past value
            ;;
        --mem_frac)
            mem_fraction_static="$2"
            shift # past argument
            shift # past value
            ;;
        *)    # unknown option
            shift # past argument
            ;;
    esac
done

# launch sglang server - takes up one whole gpu
python3 -m sglang.launch_server --model-path liuhaotian/"$model_name" --tokenizer-path "$tokenizer_name" --port "$port_number" --mem-fraction-static "$mem_fraction_static"
