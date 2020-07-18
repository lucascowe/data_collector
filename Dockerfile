FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3 python3-pip python-dev build-essential
# WORKDIR /app
COPY requirements.txt /requirements.txt
RUN python3 -m pip install -r requirements.txt
COPY app.py /app.py
COPY smp.py /smp.py
COPY key.py /key.py
COPY static /static
COPY templates /templates
# COPY data /data
COPY track_list.db /track_list.db
# COPY chromedriver /chromedriver
CMD gunicorn -b 0.0.0.0:5000 app:app