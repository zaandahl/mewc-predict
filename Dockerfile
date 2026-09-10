# Required: use the fixed parent built from the matching source lock.
ARG MEWC_FLOW_BASE
FROM ${MEWC_FLOW_BASE}
WORKDIR /code
COPY src/ .
ENV KERAS_BACKEND=tensorflow
CMD ["python", "./mewc_predict.py"]
