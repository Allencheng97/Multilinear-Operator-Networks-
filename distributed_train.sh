#!/bin/bash
NUM_PROC=$1
shift
CUDA_VISIBLE_DEVICES=2,3,4 torchrun --nproc_per_node=$NUM_PROC train.py "$@"

