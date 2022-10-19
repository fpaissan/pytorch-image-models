FROM nvcr.io/partners/gridai/pytorch-lightning:v1.4.0
ARG DEBIAN_FRONTEND=noninteractive
RUN pip install timm
WORKDIR /timm