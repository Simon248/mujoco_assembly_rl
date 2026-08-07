
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

Les STL visuels restent intacts et sont utilisés pour le rendu. La table utilise
`chandelier_assembly_table_collision.stl` pour les collisions, tandis que
`chandelier_assembly_table_visual.stl` reste réservé à l'affichage. Les
collisions sont calculées par le plugin SDF natif de MuJoCo 3.1.4 ; elles
respectent donc les cavités et concavités des pièces, sans hull convexe. L'image
Docker vérifie que la bibliothèque SDF est bien fournie par MuJoCo.

La pièce est tenue par un portique cartésien à six degrés de liberté. La tâche
active se concentre exclusivement sur son segment `place` enregistré ; les
pièces déjà assemblées sont chargées comme obstacles fixes. La commande tactile
résiduelle à sept dimensions est détaillée dans la section d'entraînement
ci-dessous.

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

L'action est
`[vx_res, vy_res, vz_res, wx_res, wy_res, wz_res, progression_residuelle]`.
Les six commandes cartésiennes résiduelles restent actives sur toute la
trajectoire : il n'existe aucun verrou ni aucune rampe d'autorité fondés sur la
progression. SAC peut donc adapter l'approche avant le premier contact ; le coût
de l'offset accumulé l'incite néanmoins à rester proche du chemin enregistré.

Le contrôleur expose trois modes à SAC dans chaque observation de l'historique :

- `tracking` : une action de progression nulle suit le chemin à sa vitesse
  nominale, `-1` l'arrête et `+1` l'accélère jusqu'à 1,5 fois cette vitesse ;
- `contact_search` : une action de progression nulle n'avance plus qu'à 25 % de
  la vitesse nominale, `-0,25` maintient `s`, une valeur plus négative recule,
  et l'avance est plafonnée à 50 % puis ralentie par les efforts mesurés ;
- `recovery` : l'augmentation de `s` est interdite, tandis que les six
  corrections résiduelles, translations et rotations, permettent le dégagement
  borné.

`contact_search` est déclenché uniquement par un contact simulé. Cette
activation tactile est mémorisée jusqu'au reset, même si le contact disparaît
ensuite. Un wrench inertiel en espace libre ne peut donc pas déclencher le
latch. Le seuil `s=0,85` sert seulement à commencer la détection d'une
stagnation près du but ; il ne change pas le mode de commande.

L'admittance n'interprète le wrench comme un effort de contact que pendant un
contact simulé courant. En espace libre, même après un latch antérieur, elle
revient vers zéro et n'infléchit donc pas la trajectoire à cause des seules
charges inertielles.

Les rotations résiduelles restent disponibles pendant `contact_search`. Après
le latch tactile, dans `contact_search` ou en `recovery`, le contrôleur autorise
24 mm de correction résiduelle et 6 mm d'admittance, soit 30 mm au total. Cette
limite élargie n'est jamais activée par la seule progression. Après un contact,
un couple supérieur à 4,5 Nm ou une force supérieure à 20 N doit persister cinq
décisions avant de déclencher une récupération, y compris en `contact_search`.
Un contact faible qui reste sous ces seuils peut demeurer en recherche tactile ; une
stagnation détectée près du but déclenche également la récupération. Seuls les
seuils durs de 8 Nm ou 80 N interrompent immédiatement la simulation. Les
limites d'admittance, d'effort et de correction sont regroupées dans
`ResidualConfig` de `src/assembly_env.py`.

Chaque trame d'observation contient aussi la progression maximale déjà atteinte,
le latch tactile, l'offset résiduel accumulé, l'offset d'admittance et sa
vitesse. SAC peut ainsi distinguer deux états ayant la même pose mesurée mais
des cibles internes différentes. Une trame contient 56 valeurs et l'historique
de huit trames en contient 448. La pose réelle pièce–gabarit reste cachée à la
politique.

La récompense dense d'alignement est bornée dans `[-0,1 ; 0,1]` et devient plus
informative près du but. Une portion du chemin ne rapporte qu'une fois, lors du
nouveau maximum de progression : reculer puis réavancer ne permet pas de gagner
la même récompense en boucle. Elle inclut aussi un coût explicite de l'offset
accumulé. Les bonus et pénalités de fin sont séparés dans `terminal_reward` :
`+250` pour un succès, `-800` pour un seuil dur, `-900` si force et couple sont
tous deux dangereux, et `-300` pour l'échec d'une récupération. À la limite de
temps, la qualité de la pose finale ajoute une pénalité bornée à `-60`. Le
facteur d'actualisation SAC est `gamma=0,999`, adapté aux 700 décisions de
l'épisode. Les pénalités de sécurité restent ainsi dominantes même lorsqu'un
seuil dur n'est atteint que très tard dans l'épisode et donc fortement actualisé.
Les troncatures à 700 décisions sont stockées comme de vraies fins d'épisode
dans le replay buffer : le critic ne prolonge pas artificiellement leur valeur
au-delà du reset suivant.

Le CSV expose notamment le déclencheur et le latch de recherche tactile, les
offsets linéaire et angulaire, ainsi que `offset_cost`, `dense_reward` et
`terminal_reward`.

Cette nouvelle sémantique de commande et l'ajout des états internes à
l'observation ne sont pas compatibles avec les anciens checkpoints : il faut
réentraîner SAC depuis zéro après cette modification.

Avant de relancer SAC, vérifier que le contrôleur nominal termine le segment
avec toutes les variabilités à zéro, sans charger de réseau et sans activer la
récupération :

```bash
ASSEMBLY_PART=part_1 NOMINAL_EPISODES=5 docker compose run --rm evaluate-nominal
```

Le rapport est écrit dans
`data/output/<pièce>/nominal_evaluation.json`. Tant que `baseline_passed` vaut
`false`, le défaut se situe dans le chemin, les collisions ou le contrôleur
nominal ; entraîner SAC ne constitue pas encore un test pertinent.

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
