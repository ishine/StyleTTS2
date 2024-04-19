
#!/bin/bash

while true; do
    torchrun train_second_ddp_no_accelerate.py --config_path Configs/config.yml
        if [ -e oom_status ]; then
        echo "OOM status detected. Re-running the script."
        rm oom_status
    else
        echo "Exiting script."
        break
    fi
done