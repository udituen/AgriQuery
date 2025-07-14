# Dockerfile for AgriQuery project

# base image for building my docker container
FROM python:3.12-slim

# for installing dependencies and libraries 
RUN pip install pandas 

# copy command copies files from my local system into the docker image
# copy from to
COPY . ./

## Add performs the same thing as the copy, but it does more by handling remote urls and unpacking compressed files
ADD ./ ./


# WOKDIR sets the working directory in the container. eg the root directory
WORKDIR /

# expose tells docker which port the container uses
EXPOSE 5000

# env sets the environment variables
ENV PROJECT_NAME='AgriQuery'

# volume sets the mount point where data is persisted in the environment outside of the container
VOLUME .



# CMD command defines what docker runs when it starts the app
CMD ['python', 'scripts/ingest.py']







