.PHONY: build train evaluate tensorboard shell clean-output

build:
	docker compose build

train:
	docker compose run --rm train

evaluate:
	docker compose run --rm evaluate

tensorboard:
	docker compose up tensorboard

shell:
	docker compose --profile tools run --rm shell

clean-output:
	find data/output -mindepth 1 ! -name .gitkeep -delete
