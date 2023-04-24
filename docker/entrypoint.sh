#!/bin/bash
mkdir -p /models; cd /models
echo 'Downloading models...'
# gdown https://drive.google.com/file/d/1-xwtKwfJZSrmJrU-_Xdkx1kPuZao7r7e/view?usp=sharing --fuzzy --continue
# gdown https://drive.google.com/file/d/1zHAhECq1uGPR9Rt1EDsTai1LbRx0jYIo/view?usp=share_link%22 --fuzzy --continue


cd /app
uvicorn serve:app --host 0.0.0.0 --port 8000
