FROM mambaorg/micromamba
MAINTAINER Ben Sherman <bentshermann@gmail.com>

RUN \
   micromamba install -y -n base -c defaults \
      matplotlib \
      numpy \
      pandas \
      python=3.7 \
      scikit-learn \
   && micromamba clean -a -y
