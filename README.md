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

## Reverse Curriculum Generation (V21)

`configs/test1V21.yaml` ajoute uniquement un curriculum de réinitialisation à
la tâche/reward V20. Une simulation MuJoCo dédiée part du goal exact, applique
des random walks 6D par la même chaîne commande–admittance–weld, puis qualifie
les starts avec la policy stochastique. Ces transitions ne passent jamais par
les workers d'apprentissage ni par le replay buffer. Les resets d'entraînement
sont dérivés du YAML : 80 % curriculum, dont 62,5 % frontier et 37,5 %
historical/mastered, soit environ 50 % frontier, 30 % historique et 20 % vrai
départ. Le historical est stratifié en quantiles de `generation_depth`; un bin
est tiré uniformément, puis un état dans ce bin. Un pool vide bascule vers
l'autre, et deux pools vides reviennent sans erreur au vrai départ. `too_hard`
n'est jamais une source de reset. Toutes les évaluations gardent de façon
persistante le vrai départ à 40 mm.

À chaque mise à jour, toute la frontier est requalifiée sur cinq rollouts, avec
des échantillons configurables de mastered et de too-hard. Les too-hard dont le
parent est actuellement mastered sont prioritaires. La fréquence de cette
revalidation est indépendante de l'expansion et vaut une fois par update par
défaut (`revalidation.every_n_curriculum_updates: 1`). Les tailles des
échantillons restent respectivement 8 et 12.

L'expansion part des feuilles du sous-graphe mastered et répartit son budget
entre plusieurs branches mélangées avec le RNG curriculum. Une branche peut
enchaîner plusieurs random walks physiques dans le même update tant que chaque
nouvel état est immédiatement classé mastered. Chaque hop repart du snapshot
exact du précédent et crée donc le lineage `A → B → C`, avec une profondeur
augmentée de un à chaque fois. Frontier, too-hard, duplicate ou état invalide
arrêtent la branche. Le nombre de hops par branche et le nombre total de
candidats qualifiés bornent strictement le coût :

```yaml
curriculum:
  expansion:
    max_hops_per_seed: 4
    max_candidates_per_update: 24
    initial_scale: 1.0
    scale_up_factor: 1.25
    scale_down_factor: 0.7
    min_scale: 0.5
    max_scale: 3.0
```

Le premier hop utilise l'amplitude de random walk existante. Après un candidat
mastered, le hop suivant multiplie cette amplitude par 1,25, dans les bornes
`min_scale` et `max_scale`. Frontier et too-hard ne déclenchent pas de seconde
tentative immédiate; `scale_down_factor` est conservé pour une adaptation
ultérieure, mais le scale n'est pas persisté dans cette version.
`expansion_scale` est uniquement un réglage
de génération reverse : il ne mesure ni la difficulté ni la progression. La
classification reste fondée uniquement sur le taux de succès courant de la
policy.

Les anciens `candidates_per_update` et `walks_per_seed` restent utilisés par le
générateur de bootstrap/diagnostic. Un update d'entraînement utilise un walk
par hop et son plafond dur est exclusivement
`expansion.max_candidates_per_update`.

`pose_distance` reste une mesure de diagnostic et peut augmenter ou diminuer
le long d'une branche. Elle ne choisit plus les seeds, les revalidations, le
historical ni le pruning. Ce dernier préserve simplement une diversité de
profondeurs et de branches, afin que le mastered reste une mémoire longue
durée.

Avant tout entraînement V21 :

```bash
CONFIG=configs/test1V21.yaml make diagnose-curriculum
CONFIG=configs/test1V21.yaml RUN_NAME=test1V21 make train
```

Pour diagnostiquer directement les pools sauvegardés (les 10 000 tirages de
sampling sont virtuels et n'avancent pas MuJoCo) :

```bash
CONFIG=configs/test1V21.yaml \
CURRICULUM_STATE=data/output/test1V21/checkpoints/curriculum_350000_steps.pkl \
make diagnose-curriculum
```

Les checkpoints V21 coordonnent modèle, replay et
`curriculum_<steps>_steps.pkl`; `curriculum_state.pkl` conserve aussi la copie
la plus récente. Une reprise dans un nouveau run peut être lancée avec
`RESUME_MODEL`, et facultativement `RESUME_REPLAY_BUFFER` / `RESUME_CURRICULUM`
si leurs noms ne suivent pas la convention du checkpoint. L'absence du replay
est une erreur (reprise SAC non fidèle); l'absence du curriculum produit un
warning puis un bootstrap propre depuis le goal.

Les paramètres du bloc `expansion` et
`revalidation.every_n_curriculum_updates` décrivent la stratégie courante : ils
peuvent être ajustés lors d'une reprise sans invalider un ancien pickle
curriculum. Les anciens YAML qui ne les déclarent pas reçoivent les valeurs par
défaut ci-dessus et une fréquence de revalidation égale à 1.

Les anciens pickles V21 minimalistes sont migrés sans régénérer les pools : la
clé `mastered` devient directement la source `curriculum_historical`, tandis
que frontier et too-hard sont conservés. Chaque ancien snapshot sans filiation
reçoit un identifiant unique et devient une racine legacy de profondeur zéro ;
les expansions suivantes reconstruisent progressivement les branches. Pour le
run V21 existant, le triplet
strictement coordonné recommandé est le modèle, le replay et le curriculum du
checkpoint 350k. Le modèle interrompu à 376 384 transitions peut fonctionner
avec le curriculum 350k, mais cette combinaison n'est pas une reprise RNG
bit-à-bit et produit donc un avertissement. Les nouveaux pickles enregistrent
le timestep SAC afin de détecter automatiquement ce décalage.

`monitor.csv` distingue `true_start`, `curriculum_frontier` et
`curriculum_historical`, avec `is_curriculum_reset` et les métriques du start.
TensorBoard sépare fractions d'épisodes et de transitions, durées et succès par
source, distances réellement utilisées et couverture min/quartiles/max des
trois pools. Il publie aussi la profondeur maximale globale et par pool, le
nombre de feuilles mastered et la profondeur des seeds d'expansion. Les
métriques d'expansion distinguent candidats, hops, branches, rollouts,
nouveaux mastered/frontier/too-hard, efficacité de découverte de la frontier
et échelles utilisées; les rollouts de revalidation sont comptés séparément.

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
