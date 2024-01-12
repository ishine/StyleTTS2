cd /root/StyleTTS2
accelerate launch --mixed_precision=fp16 train_second_segmented.py --config_path ./Configs/config_resume.yml &