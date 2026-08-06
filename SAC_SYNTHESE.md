# Synthèse SAC — entraînement et inférence

Ce document décrit le pipeline SAC : une tâche MuJoCo d'assemblage CAD de deux pièces, pilotée par Stable-Baselines3.

## Vue d'ensemble

```text
AssemblyEnv / MuJoCo → observation → politique SAC → action → MuJoCo
       ▲                                      │                  │
       └──────── replay buffer ← transition (s, a, r, s') ───────┘
```

- L'environnement est dans `src/assembly_env.py`; le modèle physique est `data/input/scene.xml`.
- `make train` lance `src/train.py` dans le service Docker `train`.
- `make evaluate` lance `src/evaluate.py` dans le service `evaluate`.
- Les résultats sont persistés sous `data/output/`.

SAC est un algorithme **off-policy** pour actions continues : les transitions sont placées dans un replay buffer, puis réutilisées pour apprendre un acteur stochastique et deux critiques. L'entropie de SAC encourage l'exploration pendant l'entraînement.

## Environnement et interface du modèle

### Action

La politique produit six valeurs normalisées dans `[-1, 1]` :

```text
[delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw]
```

Chaque action est bornée, ajoutée à une cible de commande, puis cette cible est bornée par les `ctrlrange` du MJCF. L'échelle physique par décision est :

| Composante | Maximum |
|---|---:|
| X, Y, Z | 0,8 mm |
| Roll, pitch, yaw | 0,6° chacun |

Un `step()` Gym exécute 20 pas MuJoCo (`frame_skip=20`) ; avec le timestep du scénario, la politique agit à 50 Hz.

### Observation

L'observation `float32` contient **38 valeurs** :

| Bloc | Taille | Échelle appliquée |
|---|---:|---|
| Positions articulaires | 6 | centrées et divisées par la demi-course |
| Vitesses articulaires | 6 | `/ [0.20, 0.20, 0.20, 2.0, 2.0, 2.0]` |
| Position effecteur | 3 | `/ 0.20 m` |
| Quaternion effecteur | 4 | aucune |
| Position pièce relative au repère CAD fixe | 3 | `/ 0.05 m` |
| Quaternion relatif | 4 | aucune |
| Force/couple poignet | 6 | `/ [50, 50, 50, 5, 5, 5]` |
| Action précédente | 6 | déjà normalisée |

L'ordre et les normalisations font partie du contrat du modèle : ils doivent être identiques à l'entraînement et à toute inférence future (par exemple ROS).

### Reset, objectif et fins d'épisode

Dans la distribution finale, X/Y sont tirés dans `[-0.10, 0.10] m`, la pièce démarre à `z=+0.20 m`, et roll/pitch/yaw sont tirés dans `[-15°, 15°]`. Le curriculum commence avec des poses plus proches, toujours sans contact initial. Les vitesses sont remises à zéro, puis 30 pas MuJoCo stabilisent l'état et servent à tarer le capteur F/T.

L'objectif est la pose relative `([0, 0, 0], quaternion identité)` du repère CAD partagé. Un épisode dure au plus 250 décisions et s'arrête aussi en cas de :

- succès : erreur de position < 1,5 mm et rotation < 2° ;
- contact dangereux : norme de force > 120 N.

Les poses de départ sont échantillonnées avec rejet des contacts pièce-table.
Un curriculum mélange progressivement la distribution finale à des poses plus
proches, sans ajouter de waypoint ni contraindre la trajectoire de la politique.

### Récompense

```text
250 × diminution de l'erreur de position
+ 0.5 × diminution de l'erreur angulaire
- 0.01 × ||action précédente||²
- 0.0005 × max(norme_force - 15 N, 0)²
+ 0.05 près de l'axe (erreur latérale < 2.5 mm)
+ 0.10 à proximité de la pose d'assemblage (z relatif < 15 mm)
+ 20 en cas de succès, ou -10 si la force est dangereuse
```

Les échelles d'action, la randomisation initiale, la récompense et les seuils de succès sont les leviers qui changent le plus la politique apprise.

## Entraînement SAC

### Hyperparamètres explicitement fixés

| Paramètre | Valeur | Effet |
|---|---:|---|
| Politique | `MlpPolicy` | réseaux entièrement connectés |
| `net_arch` | `[64, 64]` | deux couches cachées de 64 neurones |
| `learning_rate` | `3e-4` | vitesse d'optimisation |
| `gamma` | `0.99` | importance des récompenses futures |
| `tau` | `0.005` | mise à jour douce des cibles critiques |
| `train_freq` | `1` | entraînement après chaque transition |
| `gradient_steps` | `1` | une mise à jour par déclenchement |
| `BUFFER_SIZE` | `50_000` | capacité du replay buffer |
| `BATCH_SIZE` | `256` | transitions par mise à jour |
| `LEARNING_STARTS` | `5_000` | remplissage avant apprentissage |
| `TOTAL_TIMESTEPS` | `500_000` | budget par défaut |
| `SEED` | `7` | reproductibilité partielle |

Les paramètres non passés à `SAC(...)` (dont le réglage d'entropie) restent aux valeurs par défaut de la version installée de Stable-Baselines3.

### Déroulement

1. `check_env()` vérifie l'environnement Gymnasium (sauf `--skip-env-check`).
2. `Monitor` journalise les épisodes et les métriques `info`.
3. Les `LEARNING_STARTS` premières transitions remplissent le buffer ; aucune mise à jour de réseau n'est alors réalisée.
4. Ensuite, chaque transition déclenche un échantillonnage de 256 expériences et une mise à jour acteur / critiques.
5. Un checkpoint est écrit toutes les `CHECKPOINT_FREQ=50_000` décisions.
6. Le modèle final est sauvegardé dans `data/output/models/assembly_sac.zip`; `Ctrl+C` sauvegarde plutôt `assembly_sac_interrupted.zip`.

Les checkpoints ne contiennent pas le replay buffer (`save_replay_buffer=False`) : ils sont adaptés à l'évaluation, mais ne permettent pas une reprise exactement identique de l'entraînement.

### Commandes et GPU

```bash
TOTAL_TIMESTEPS=500000 LEARNING_STARTS=5000 \
BUFFER_SIZE=50000 BATCH_SIZE=256 CHECKPOINT_FREQ=50000 \
SEED=7 SB3_DEVICE=cuda make train
```

Le service `train` expose toutes les GPU et `SB3_DEVICE=cuda` est son défaut. Avec le petit MLP `[64, 64]`, une occupation GPU faible est normale : MuJoCo, les interactions Python et la simulation peuvent être limitants.

### Sorties et métriques à suivre

| Emplacement | Contenu |
|---|---|
| `data/output/models/` | modèle final SAC |
| `data/output/checkpoints/` | modèles intermédiaires |
| `data/output/monitor/train.monitor.csv` | retours, longueur et métriques d'épisode |
| `data/output/tensorboard/` | pertes, retours et statistiques SB3 |
| `data/output/training_metadata.json` | modèle, seed, pas demandés/réalisés, complétion |

Suivre en priorité le taux de succès, l'erreur latérale/position, la force, le retour moyen et les pertes acteur/critique. Une hausse du retour seule ne garantit pas une insertion : la récompense contient des bonus de progression.

## Évaluation et inférence

L'évaluation recrée le même environnement, charge le fichier final ou, à défaut, le `.zip` le plus récent dans `models/` ou `checkpoints/`. La politique est appelée de façon déterministe :

```python
action, _ = model.predict(observation, deterministic=True)
```

Les poses initiales restent aléatoires. `EVAL_SEED=100` et `EVAL_EPISODES=10` utilisent donc les graines 100 à 109 et rendent l'essai reproductible pour un même code, XML et modèle.

```bash
EVAL_EPISODES=25 EVAL_SEED=100 make evaluate
```

`data/output/evaluation.json` contient le modèle et XML employés, le nombre et taux de succès, récompense/pas moyens, puis le détail de chaque épisode (graine, succès et métriques finales). Pour contrôler strictement un essai, définir `MODEL_PATH` plutôt que de dépendre de la sélection automatique du dernier fichier.

La visualisation :

```bash
EVAL_RENDER=human make evaluate-gui
```

Elle cadence la simulation en temps réel : elle ne convient pas pour mesurer le débit.

### Périphérique d'inférence

Le service `evaluate` expose CUDA, mais `evaluate.py` charge actuellement avec `SAC.load(...)` sans argument `device`. `SB3_DEVICE` est donc explicitement utilisé par l'entraînement, pas par ce chargement. Stable-Baselines3 choisit alors son comportement `auto`. Pour imposer et tracer le périphérique, il faudrait charger avec `device=os.environ.get("SB3_DEVICE", "auto")` et afficher le device retenu.

## Vigilances importantes

- Les valeurs initiales actuelles ±0,2 m dépassent les `ctrlrange` X/Y ±0,025 m du MJCF. La commande est bornée, mais certains départs peuvent être hors de portée : confirmer que cette difficulté est voulue avant un long entraînement.
- Ne pas modifier observation, action, unités ou normalisation entre entraînement et inférence, sauf pour tester explicitement la généralisation.
- Pour comparer des checkpoints, conserver XML, seeds et épisodes identiques, puis archiver l'`evaluation.json` correspondant.
- Agrandir le réseau, le batch ou paralléliser des environnements peut augmenter l'usage GPU, mais modifie aussi stabilité, débit et qualité : ce n'est pas un gain automatique.
