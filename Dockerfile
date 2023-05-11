FROM python:3.9
COPY . /app
WORKDIR /app/
# RUN apt update
# RUN apt install -y python3.9 python3.9-dev python3.9-venv python3-pip python3-wheel build-essential
RUN python3.9 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
