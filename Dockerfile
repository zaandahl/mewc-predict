# set base image (host OS)
FROM zaandahl/mewc_flow:latest

# set the working directory in the container
WORKDIR /code

# copy the src to the working directory
COPY mewc_predict/src/ .
COPY mewc_common.py .
COPY config.yaml .

# run en_predict on start
CMD [ "python", "./en_predict.py" ]