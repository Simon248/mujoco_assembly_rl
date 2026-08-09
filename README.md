# Tenon–mortaise avec SAC dans MuJoCo

Environnement minimal de peg-in-hole, sans trajectoire nominale ni Residual RL.
La politique SAC commande un incrément cartésien 6D dans le repère configuré.
`action.action_frame: task` représente un delta du repère CAD mobile exprimé
dans les axes du CAD cible, converti analytiquement en cible grasp; `grasp`
conserve le comportement historique.
Une archive dépourvue de ce paramètre utilise `grasp` par compatibilité.
Le contrôleur d'admittance transforme ensuite le wrench de contact avant de
déplacer le tenon physiquement dans MuJoCo.

```text
pose relative observée + wrench(grasp) -> SAC -> delta 6D -> admittance -> MuJoCo
                                                        ^                 |
                                                        +-----------------+
```

La convention unique est `T_fixed_to_mobile`: pose du repère CAD du tenon dans
celui de la mortaise. L'erreur, réelle ou observée, vaut
`T_target^-1 @ T_fixed_to_mobile`. La cible est déclarée dans le YAML, même si
les CAD actuels utilisent l'identité.

## Démarrage (uniquement Docker)

```bash
make build
make test
make debug                         # Test 0, repères MuJoCo et commande manuelle
CONFIG=configs/test1.yaml RUN_NAME=test1_v2 make train
RUN_NAME=test1_v2 make evaluate
RUN_NAME=test1_v2 EVAL_EPISODES=1 make evaluate-gui  # viewer MuJoCo via X11
# Lecture au quart de la vitesse réelle :
RUN_NAME=test1_v2 EVAL_EPISODES=1 RENDER_SPEED=0.25 make evaluate-gui
make tensorboard                   # http://localhost:6006
```

`make debug` et `make evaluate-gui` utilisent X11 et ouvrent le viewer MuJoCo. Selon l'hôte, il peut être
nécessaire d'autoriser Docker temporairement avec `xhost +local:root`.

Chaque entraînement est isolé dans `data/output/<RUN_NAME>/`. Outre le modèle,
les checkpoints et les métriques, le run archive le YAML résolu, les versions,
les empreintes des CAD/grasp poses et un snapshot du code source. La gestion
des commits et du versionnement Git reste volontairement externe au pipeline.

Pour comparer explicitement deux checkpoints sans mélanger leurs résultats :

```bash
RUN_NAME=test1_v2 MODEL_PATH=checkpoints/sac_50000_steps.zip make evaluate
RUN_NAME=test1_v2 MODEL_PATH=checkpoints/sac_100000_steps.zip make evaluate
# Liste explicite ou tous les checkpoints, toujours avec deterministic=True :
RUN_NAME=test1_v2 MODEL_PATHS=checkpoints/sac_50000_steps.zip,checkpoints/sac_100000_steps.zip make evaluate
RUN_NAME=test1_v2 EVAL_ALL_CHECKPOINTS=1 EVAL_EPISODES=1 make evaluate
# Trajectoire détaillée du premier épisode uniquement sur demande :
RUN_NAME=test1_v2 MODEL_PATH=checkpoints/sac_100000_steps.zip EVAL_TRAJECTORY=1 make evaluate
```

Les fichiers sont écrits sous `evaluations/<nom_du_modèle>_episodes.csv`,
`evaluations/<nom_du_modèle>_summary.json` et `evaluations/checkpoints_summary.csv`,
avec le SHA-256 du modèle. Le fichier `_trajectory.csv` est optionnel. Lorsque
la configuration ne contient aucune randomisation, plusieurs épisodes sont des
répétitions du même scénario et le résumé le signale explicitement.

## Environnements parallèles

Le parallélisme est configuré dans le YAML du run :

```yaml
training:
  n_envs: 4
  base_seed: 7
  checkpoint_freq: 50000
```

`n_envs: 1` utilise un `DummyVecEnv`; les valeurs supérieures utilisent un
`SubprocVecEnv` avec une simulation MuJoCo indépendante par processus. Les
workers reçoivent les seeds `base_seed + rank`. `VecMonitor` reste dans le
processus principal et écrit un unique `monitor.csv`.

SAC collecte un pas vectorisé puis effectue une mise à jour par transition
collectée (`train_freq=(1, "step")`, `gradient_steps=-1`). Les fréquences de
checkpoint restent exprimées en transitions, indépendamment de `n_envs`.

## Fichiers importants

- [configs/test0.yaml](configs/test0.yaml) contient tous les paramètres
  physiques, reward, bruit et randomisation; `test1` à `test4` activent les
  difficultés progressivement.
- [src/assembly_env.py](src/assembly_env.py) construit la scène CAD et expose
  l'observation `[erreur pose 6D, wrench 6D, offset de position de
  l'admittance 6D]`. L'offset est normalisé par `admittance.max_offset`; il est
  connu du contrôleur réel et aucune vitesse MuJoCo ou interne n'est exposée.
- [src/wrench.py](src/wrench.py) somme les forces de contact et transporte les
  couples au grasp frame; ce n'est pas un simple capteur de site.
- [src/admittance.py](src/admittance.py) est testable indépendamment du RL.
- [src/task_logic.py](src/task_logic.py) impose un succès sûr, les raisons de
  terminaison et la pénalité terminale de sécurité.
- [src/debug.py](src/debug.py) est la validation mécanique obligatoire avant
  l'entraînement.

Les STL sont déjà en mètres, tandis que la grasp pose source est en millimètres.
La conversion `.001` s'applique donc uniquement à la position du grasp dans
`assembly_env.py`.
Les rotations sont toujours composées sous forme de quaternion/rotation-vector,
jamais en ajoutant du bruit à un quaternion.

Dans `monitor.csv`, les erreurs par axe sont en mètres/radians et les colonnes
`action_*` contiennent la dernière sortie SAC acceptée, normalisée dans `[-1, 1]`,
dans le repère `action_frame`. La trajectoire d'évaluation optionnelle contient
ces mêmes valeurs à chaque pas.

Les seuils de force/couple sont contrôlés à chaque sous-pas MuJoCo. Un état
géométriquement assemblé ne compte comme succès que s'il respecte simultanément
les limites de sécurité. Les colonnes `episode_reward_*` du monitoring sont les
sommes par épisode; `reward_*` reste la valeur du dernier pas pour le diagnostic.
