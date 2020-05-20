FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3 python3-pip python-dev build-essential
#COPY . /app
WORKDIR /app
RUN python3 -m pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]