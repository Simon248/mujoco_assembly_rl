# Tenon–mortaise avec SAC dans MuJoCo

Environnement minimal de peg-in-hole, sans trajectoire nominale ni Residual RL.
La politique SAC commande directement un incrément cartésien 6D du `grasp_frame`.
Le contrôleur d'admittance transforme cette consigne et le wrench de contact avant
de déplacer le tenon physiquement dans MuJoCo.

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
make debug                         # Test 0, repères MuJoCo et commande manuelle
CONFIG=configs/test1.yaml RUN_NAME=test1 make train
RUN_NAME=test1 make evaluate
RUN_NAME=test1 EVAL_EPISODES=1 make evaluate-gui  # viewer MuJoCo via X11
# Lecture au quart de la vitesse réelle :
RUN_NAME=test1 EVAL_EPISODES=1 RENDER_SPEED=0.25 make evaluate-gui
make tensorboard                   # http://localhost:6006
```

`make debug` et `make evaluate-gui` utilisent X11 et ouvrent le viewer MuJoCo. Selon l'hôte, il peut être
nécessaire d'autoriser Docker temporairement avec `xhost +local:root`.

Chaque entraînement est isolé dans `data/output/<RUN_NAME>/`: `config.yaml`,
`monitor.csv`, événements TensorBoard, checkpoints, modèle final et résultats
d'évaluation CSV/JSON.

## Fichiers importants

- [configs/test0.yaml](configs/test0.yaml) contient tous les paramètres
  physiques, reward, bruit et randomisation; `test1` à `test4` activent les
  difficultés progressivement.
- [src/assembly_env.py](src/assembly_env.py) construit la scène CAD et expose
  l'observation minimale `[erreur pose 6D, wrench 6D]`.
- [src/wrench.py](src/wrench.py) somme les forces de contact et transporte les
  couples au grasp frame; ce n'est pas un simple capteur de site.
- [src/admittance.py](src/admittance.py) est testable indépendamment du RL.
- [src/debug.py](src/debug.py) est la validation mécanique obligatoire avant
  l'entraînement.

Les STL sont déjà en mètres, tandis que la grasp pose source est en millimètres.
La conversion `.001` s'applique donc uniquement à la position du grasp dans
`assembly_env.py`.
Les rotations sont toujours composées sous forme de quaternion/rotation-vector,
jamais en ajoutant du bruit à un quaternion.
