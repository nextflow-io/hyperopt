version ?= v1.0

all: build push

build:
	cp ../conda.yml .
	docker build --output=type=docker --progress=plain --tag quay.io/nextflow/ml-hyperopt:${version} .
	rm conda.yml

push:
	docker push quay.io/nextflow/ml-hyperopt:${version}