
#!/bin/bash

while true; do
    accelerate launch --mixed_precision=fp16 train_second_ddp.py --config_path ./Configs/config_resume.yml
    if [ -e oom_status ]; then
        echo "OOM status detected. Re-running the script."
        rm oom_status
    else
        echo "Exiting script."
        break
    fi
done