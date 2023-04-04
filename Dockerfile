FROM armswdev/armswdev/pytorch-arm-neoverse:r23.03-torch-1.13.0-onednn-acl

RUN sudo apt-get update && sudo apt-get -y upgrade
RUN sudo apt-get install -y libsm6 libxext6 libxrender-dev
RUN sudo apt-get install -y ffmpeg
RUN pip install --upgrade pip
RUN pip install Pillow fastapi pydantic uvicorn python-multipart backgroundremover opencv-python-headless
RUN pip install --upgrade numpy==1.23.5
WORKDIR /
COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]