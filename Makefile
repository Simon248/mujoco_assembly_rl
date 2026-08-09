.PHONY: build test train evaluate evaluate-gui evaluate-scripted diagnose-sac-q diagnose-sac-branching debug tensorboard
build:
	docker compose build
test:
	docker compose run --rm --no-deps train python -m unittest discover -s tests -v
train:
	docker compose run --rm train
# CONFIG=configs/test1V8.yaml RUN_NAME=test1V10 make train
evaluate:
	docker compose run --rm --remove-orphans evaluate
evaluate-scripted:
	docker compose run --rm --no-deps train python -m src.evaluate_scripted --config $${CONFIG:-configs/test1V14.yaml} --episodes $${EVAL_EPISODES:-20}
diagnose-sac-q:
	docker compose run --rm --no-deps train python -m src.diagnose_sac_q --run data/output/$${RUN_NAME:-test1V14} $${MODEL_PATH:+--model $$MODEL_PATH}
diagnose-sac-branching:
	docker compose run --rm --no-deps train python -m src.diagnose_sac_branching --run data/output/$${RUN_NAME:-test1V14} $${MODEL_PATH:+--model $$MODEL_PATH}
evaluate-gui:
	xhost +local:root >/dev/null && docker compose run --rm --remove-orphans evaluate-gui; status=$$?; xhost -local:root >/dev/null; exit $$status

# RUN_NAME=test1 EVAL_EPISODES=1 RENDER_SPEED=0.25 make evaluate-gui
# RUN_NAME=test1V2 EVAL_EPISODES=20 RENDER_SPEED=0.25 make evaluate-gui

debug:
	xhost +local:root >/dev/null && docker compose run --rm debug; xhost -local:root >/dev/null
tensorboard:
	docker compose up tensorboard
