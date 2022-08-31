version ?= v1.0

all: build push

build:
	docker build --output=type=docker --progress=plain --tag quay.io/nextflow/ml-example:${version} .

push:
	docker push quay.io/nextflow/ml-example:${version}