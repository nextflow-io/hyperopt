FROM mambaorg/micromamba:0.25
MAINTAINER Ben Sherman <bentshermann@gmail.com>

COPY conda.yml .
RUN \
    micromamba install -y -n base -f conda.yml && \
    micromamba clean -a -y
