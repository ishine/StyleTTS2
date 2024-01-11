FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN DEBIAN_FRONTEND=noninteractive apt update && apt install -y git ffmpeg python3-pip htop nvtop
ADD "https://api.github.com/repos/effusiveperiscope/StyleTTS2/commits?per_page=1" latest_commit
RUN git clone https://github.com/effusiveperiscope/StyleTTS2 /root/StyleTTS2
WORKDIR /root/StyleTTS2
RUN pip install -r requirements.txt gdown tqdm pyyaml requests ffmpeg-python ngrok
RUN  curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok
