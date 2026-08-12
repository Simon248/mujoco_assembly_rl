.PHONY: build test train evaluate evaluate-gui evaluate-scripted diagnose-sac-q diagnose-sac-branching diagnose-curriculum debug tensorboard
build:
	docker compose build
test:
	docker compose run --rm --no-deps train python -m unittest discover -s tests -v
train:
	docker compose run --rm train
# CONFIG=configs/test1V41.yaml RUN_NAME=test1V41 make train
evaluate:
	docker compose run --rm --remove-orphans evaluate
evaluate-scripted:
	docker compose run --rm --no-deps train python -m src.evaluate_scripted --config $${CONFIG:-configs/test1V14.yaml} --episodes $${EVAL_EPISODES:-20}
diagnose-sac-q:
	docker compose run --rm --no-deps train python -m src.diagnose_sac_q --run data/output/$${RUN_NAME:-test1V14} $${MODEL_PATH:+--model $$MODEL_PATH}
diagnose-sac-branching:
	docker compose run --rm --no-deps train python -m src.diagnose_sac_branching --run data/output/$${RUN_NAME:-test1V14} $${MODEL_PATH:+--model $$MODEL_PATH}
diagnose-curriculum:
	docker compose run --rm --no-deps train python -m src.diagnose_curriculum --config $${CONFIG:-configs/test1V21.yaml} $${CURRICULUM_STATE:+--curriculum-state $$CURRICULUM_STATE}
evaluate-gui:
	xhost +local:root >/dev/null && docker compose run --rm --remove-orphans evaluate-gui; status=$$?; xhost -local:root >/dev/null; exit $$status

# RUN_NAME=test1 EVAL_EPISODES=1 RENDER_SPEED=0.25 make evaluate-gui
# RUN_NAME=test1V2 EVAL_EPISODES=20 RENDER_SPEED=0.25 make evaluate-gui

debug:
	xhost +local:root >/dev/null && docker compose run --rm debug; xhost -local:root >/dev/null
tensorboard:
	docker compose up tensorboard


# CONFIG=configs/test1V42.yaml \
# RUN_NAME=test1V41-resume-2.7M \
# RESUME_MODEL=data/output/test1V41/model_interrupted.zip \
# make train


