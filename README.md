# MuJoCo Assembly RL — version Docker

Ce projet entraîne une politique SAC sur une tâche minimale d'insertion
pion-logement dans MuJoCo, sans ROS 2 pendant l'entraînement.

## Architecture des volumes

```text
mujoco_assembly_rl_docker/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── src/                     # monté dans /workspace/src
│   ├── assembly_env.py
│   ├── cartesian_ik.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   ├── input/               # monté en lecture seule dans /data/input
│   │   └── scene.xml
│   └── output/              # monté en lecture/écriture dans /data/output
└── .env.example
```

Le code est un bind mount : une modification locale de `src/` est visible dans
le conteneur sans reconstruire l'image. Le modèle MuJoCo est lu depuis
`data/input/`. Les modèles SAC, checkpoints, métriques et logs TensorBoard sont
écrits dans `data/output/`.

## Prérequis

- Docker Engine avec le plugin Docker Compose ;
- sous Linux, un accès en écriture à `data/output/`.

Pour éviter les fichiers appartenant à `root` sous Linux :

```bash
cp .env.example .env
sed -i "s/^LOCAL_UID=.*/LOCAL_UID=$(id -u)/" .env
sed -i "s/^LOCAL_GID=.*/LOCAL_GID=$(id -g)/" .env
```

Sur Docker Desktop, les valeurs par défaut conviennent généralement.

## Construction

```bash
docker compose build
```

ou :

```bash
make build
```

## Entraînement

```bash
docker compose run --rm train
```

Pour un test court :

```bash
TOTAL_TIMESTEPS=10000 CHECKPOINT_FREQ=5000 docker compose run --rm train
```

Résultats créés :

```text
data/output/
├── models/assembly_sac.zip
├── checkpoints/
├── monitor/train.monitor.csv
├── tensorboard/
└── training_metadata.json
```

Les variables principales peuvent être définies dans `.env` :

```dotenv
TOTAL_TIMESTEPS=500000
SEED=7
CHECKPOINT_FREQ=50000
```

## Évaluation sans interface graphique

```bash
docker compose run --rm evaluate
```

Le résumé détaillé est enregistré dans :

```text
data/output/evaluation.json
```

Pour modifier le nombre d'épisodes :

```bash
EVAL_EPISODES=25 docker compose run --rm evaluate
```

## TensorBoard

Dans un terminal :

```bash
docker compose up tensorboard
```

Puis ouvrir `http://localhost:6006`. Le port peut être changé avec
`TENSORBOARD_PORT` dans `.env`.

## Visualisation MuJoCo sous Linux/X11

L'entraînement et l'évaluation standard sont headless avec OSMesa. Pour lancer
le viewer MuJoCo sur un poste Linux utilisant X11 :

```bash
xhost +local:docker
docker compose --profile gui run --rm evaluate-gui
xhost -local:docker
```

Cette configuration graphique n'est pas directement portable vers Wayland,
macOS ou Windows. Sur ces plateformes, l'évaluation headless reste disponible.

## Shell de développement

```bash
docker compose --profile tools run --rm shell
```

Le shell voit :

```text
/workspace/src   -> ./src
/data/input      -> ./data/input, lecture seule
/data/output     -> ./data/output
```

## Changer le modèle MuJoCo

Remplacer :

```text
data/input/scene.xml
```

Le chemin peut aussi être remplacé au niveau du conteneur avec la variable
`MUJOCO_XML_PATH`, à condition que le fichier soit accessible depuis un volume.

## Passage au vrai robot

La politique produit actuellement :

```text
[delta_x, delta_y, delta_z, delta_yaw]
```

Pour un bras réel :

1. remplacer le portique cartésien dans `data/input/scene.xml` par le MJCF du robot ;
2. passer à six actions `[dx, dy, dz, dRx, dRy, dRz]` ;
3. utiliser `differential_ik_position_target()` dans `src/cartesian_ik.py` ;
4. conserver exactement l'ordre, les unités et la normalisation des observations ;
5. exporter ensuite `data/output/models/assembly_sac.zip` vers le nœud ROS 2 d'inférence.

ROS 2 n'est donc pas inclus dans ce conteneur d'entraînement.
