.PHONY: build train evaluate evaluate-gui debug tensorboard
build:
	docker compose build
train:
	docker compose run --rm train
evaluate:
	docker compose run --rm --remove-orphans evaluate
evaluate-gui:
	xhost +local:root >/dev/null && docker compose run --rm --remove-orphans evaluate-gui; status=$$?; xhost -local:root >/dev/null; exit $$status

# RUN_NAME=test1 EVAL_EPISODES=1 RENDER_SPEED=0.25 make evaluate-gui

debug:
	xhost +local:root >/dev/null && docker compose run --rm debug; xhost -local:root >/dev/null
tensorboard:
	docker compose up tensorboard
