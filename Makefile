.PHONY: build train evaluate evaluate-gui evaluate-shell tensorboard shell clean-output

build:
	docker compose build

train:
	docker compose run --rm train

evaluate:
	docker compose run --rm evaluate

# Évaluation avec fenêtre MuJoCo (GUI) affichée sur l'hôte via X11.
# Autorise l'accès au serveur X, puis lance l'évaluation en mode render=human.
evaluate-gui:
	xhost +local:root >/dev/null
	EVAL_RENDER=human MUJOCO_GL=glfw docker compose run --rm evaluate; \
		status=$$?; \
		xhost -local:root >/dev/null; \
		exit $$status

# Ouvre un shell interactif dans le conteneur evaluate,
# avec l'accès X11 configuré (pratique pour lancer manuellement
# `python -m src.evaluate --render=human`).
evaluate-shell:
	xhost +local:root >/dev/null
	MUJOCO_GL=glfw docker compose run --rm --entrypoint bash evaluate; \
		status=$$?; \
		xhost -local:root >/dev/null; \
		exit $$status

tensorboard:
	docker compose up tensorboard

shell:
	docker compose --profile tools run --rm shell

clean-output:
	find data/output -mindepth 1 ! -name .gitkeep -delete

