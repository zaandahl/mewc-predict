# set base image (host OS)
FROM zaandahl/mewc-flow:2.0.3

# set the working directory in the container
WORKDIR /code

# copy the src to the working directory
COPY src/ .

# run en_predict on start
CMD [ "python", "./mewc_predict.py" ]