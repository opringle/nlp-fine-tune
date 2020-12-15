FROM nvidia/cuda:10.2-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.8-distutils \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 10 \
    && rm -rf /var/lib/apt/lists/*

# Install latest pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    pip install setuptools && \
    rm get-pip.py

WORKDIR /root

# Install python package requirements
COPY requirements.txt /root/requirements.txt
RUN pip install -r requirements.txt

# Install and configure google cloud sdk
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
    --path-update=false --bash-completion=false \
    --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copy all training code to container
RUN mkdir /root/trainer
ADD trainer /root/trainer/

RUN python -V

# Set the entry point to invoke the training application
ENTRYPOINT ["python", "-m", "trainer.task"]