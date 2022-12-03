FROM nvcr.io/nvidia/pytorch:21.03-py3

WORKDIR /usr/src/app

SHELL ["/bin/bash", "--login", "-c"]

COPY ./requirements.txt .

RUN apt-get update && apt-get install -y tmux
RUN pip install -r requirements.txt
RUN conda init bash

CMD ["bash"]