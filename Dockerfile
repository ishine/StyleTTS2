FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN DEBIAN_FRONTEND=noninteractive apt update && apt install -y git ffmpeg python3-pip
ADD "https://api.github.com/repos/effusiveperiscope/StyleTTS2/commits?per_page=1" latest_commit
RUN git clone https://github.com/effusiveperiscope/StyleTTS2 /StyleTTS2
WORKDIR /StyleTTS2
RUN pip install -r requirements.txt gdown tqdm pyyaml requests
