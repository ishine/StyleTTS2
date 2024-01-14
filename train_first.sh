#!/bin/bash

while true; do
    accelerate launch --mixed_precision=fp16 train_first.py --config_path ./Configs/config.yml
    if [ -e oom_status ]; then
        echo "OOM status detected. Re-running the script."
        rm oom_status
    else
        echo "Exiting script."
        break
    fi
done