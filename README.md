
# usefull cmd  
```ASSEMBLY_PART=part_1 TOTAL_TIMESTEPS=150000 LEARNING_STARTS=5000 docker compose run --rm train```  
```ASSEMBLY_PART=part_3 docker compose run --rm evaluate```  

# MuJoCo Assembly RL — version Docker

Ce projet entraîne une politique SAC sur l'assemblage CAD de deux pièces dans
MuJoCo, sans ROS 2 pendant l'entraînement. `chandelier_part_1.stl` est la
pièce mobile et `chandelier_assembly_table_visual.stl` la structure fixe.

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
│   │   ├── scene.xml
│   │   └── cad/              # STL d'origine, dans le repère d'assemblage
│   └── output/              # monté en lecture/écriture dans /data/output
└── .env.example
```

Le code est un bind mount : une modification locale de `src/` est visible dans
le conteneur sans reconstruire l'image. Le modèle MuJoCo est lu depuis
`data/input/`. Les modèles SAC, checkpoints, métriques et logs TensorBoard sont
écrits dans `data/output/`.

## Collisions CAD

Les STL restent intacts et sont utilisés pour le rendu. Les collisions sont
calculées par le plugin SDF natif de MuJoCo 3.1.4, à partir de ces mêmes STL :
elles respectent donc les cavités et concavités des pièces, sans hull convexe.
L'image Docker vérifie que la bibliothèque SDF est bien fournie par MuJoCo.

La pièce est tenue par un portique cartésien à six degrés de liberté. L'action
de la politique est `[dx, dy, dz, droll, dpitch, dyaw]`. La distribution finale
de reset est `z=+0,20 m`, X/Y uniformes dans ±0,10 m et roll/pitch/yaw dans
±15°. Le succès correspond à la pose CAD relative identité, à 1,5 mm et 2° près.

L'entraînement applique un curriculum non directionnel : les poses initiales
restent toujours validées sans contact pièce-table, puis la proportion de poses
tirées dans la distribution finale augmente de 20 % à 100 % entre 0 et 350 000
décisions. Cela facilite l'exploration sans imposer de trajectoire.

Par défaut, 25 % des épisodes d'entraînement sont des épisodes de
désassemblage : la pièce commence dans la pose CAD assemblée et doit rejoindre
une pose libre aléatoire, explicitement fournie à la politique dans l'observation.

## Diagnostiquer la collision SDF

Pour exporter en PLY la surface SDF locale effectivement utilisée par MuJoCo :

```bash
docker compose run --rm train python -m src.export_sdf_pointcloud
```

Le fichier `data/output/sdf/table_isosurface_local.ply` peut être ouvert dans
MeshLab ou CloudCompare. La résolution par défaut est 2 mm ; utiliser
`--resolution 0.005` pour un premier diagnostic plus rapide.

## Tester les trajectoires et collisions

Le script rejoue les poses des trois YAML dans un MJCF de test, avec les mêmes
collisions SDF que l'environnement. Il contrôle chaque pièce contre la table et
les pièces posées avant elle. Il conserve également toutes les métadonnées YAML
dans le rapport.

```bash
docker compose run --rm train python -m src.test_env_colision
```

Le résultat est enregistré dans `data/output/collision_report.json`. Les
segments à contrôler se choisissent directement dans
`src/test_env_colision.py`, via :

```python
SELECTED_SEGMENTS = ("approach", "place")
```

Les valeurs possibles sont `approach`, `place`, `retreat` (ou `retrait`). Pour
ouvrir la relecture visuelle sous Linux/X11 :

```bash
xhost +local:docker
docker compose --profile gui run --rm test-env-colision
xhost -local:docker
```

Le bloc `summary` en tête du JSON, et la sortie terminal, donnent directement
le verdict par pièce et segment. En cas de collision, ils indiquent l'obstacle,
l'instant, la pénétration maximale et le point de contact dans le repère de la
table. Dans le viewer, les contacts déjà rencontrés sont marqués en rouge ; le
contact le plus profond est jaune. À l'entrée dans chaque contact distinct, la
relecture attend une pression sur Entrée avant de continuer. Les réglages de
vitesse, confirmation et taille des marqueurs sont en tête du script.

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

## Entraînement residual RL tactile

L'environnement entraîne désormais un residual SAC sur le segment `place`
d'une pièce donnée. La trajectoire YAML devient un chemin géométrique : la
politique choisit une progression (avance, arrêt ou recul) et une vitesse
cartésienne résiduelle. Les efforts traversent une admittance fixe ; les
erreurs de prise et de gabarit sont physiques mais cachées à la politique.

Choisir la pièce avec `ASSEMBLY_PART` (`part_1`, `part_2` ou `part_3`). Les
modèles et rapports sont enregistrés sous `data/output/<pièce>/`.

```bash
ASSEMBLY_PART=part_2 TOTAL_TIMESTEPS=500000 docker compose run --rm train
ASSEMBLY_PART=part_2 docker compose run --rm evaluate
```

L'action est `[vx_res, vy_res, vz_res, wx_res, wy_res, wz_res, progression]`.
Les limites d'admittance, de force et de correction sont regroupées dans
`ResidualConfig` de `src/assembly_env.py`.

L'évaluation affiche maintenant par défaut le viewer **MuJoCo** (le projet
n'utilise pas Gazebo). Pour une exécution sans fenêtre, notamment sur une
machine sans serveur X11 :

```bash
EVAL_RENDER=none docker compose run --rm evaluate
```

Le prétraitement SDF des STL est fait au chargement d'un modèle MuJoCo. Le
plugin `mujoco.sdf.sdflib` fourni par MuJoCo 3.1.4 ne fournit pas de cache SDF
persistant sur disque ; monter `data/output` ne peut donc pas éviter ce calcul
entre deux conteneurs. En entraînement, la validation Gym crée un environnement
supplémentaire. Après une première validation réussie, elle peut être évitée :

```bash
SKIP_ENV_CHECK=1 docker compose run --rm train
```

## Entraînement (commandes)

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

La politique produit :

```text
[delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw]
```

Pour un bras réel :

1. remplacer le portique cartésien dans `data/input/scene.xml` par le MJCF du robot ;
2. conserver ou adapter les six actions `[dx, dy, dz, dRx, dRy, dRz]` ;
3. utiliser `differential_ik_position_target()` dans `src/cartesian_ik.py` ;
4. conserver exactement l'ordre, les unités et la normalisation des observations ;
5. exporter ensuite `data/output/models/assembly_sac.zip` vers le nœud ROS 2 d'inférence.

ROS 2 n'est donc pas inclus dans ce conteneur d'entraînement.
