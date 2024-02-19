FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN DEBIAN_FRONTEND=noninteractive apt update && apt install -y git ffmpeg python3-pip htop nvtop curl
ADD "https://api.github.com/repos/effusiveperiscope/StyleTTS2/commits?per_page=1" latest_commit
RUN mkdir /root/StyleTTS2
WORKDIR /root/StyleTTS2

EXPOSE 6006 6007

Copy requirements.txt /root/StyleTTS2/

RUN pip install -r requirements.txt gdown tqdm pyyaml requests ffmpeg-python

COPY styletts2_train_remote.ipynb /root/StyleTTS2/
COPY *.py /root/StyleTTS2/
COPY *.sh /root/StyleTTS2/
Copy Configs /root/StyleTTS2/
Copy Data /root/StyleTTS2/
Copy Modules /root/StyleTTS2/
Copy Utils /root/StyleTTS2/