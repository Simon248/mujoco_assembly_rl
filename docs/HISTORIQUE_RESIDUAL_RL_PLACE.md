# Historique du residual RL tactile — segment place

État de la discussion et du dépôt au **7 août 2026**.

Ce document retrace les décisions, les implémentations, les essais et les
hypothèses invalidées pendant le développement. Il faut lire les chiffres avec
une précaution importante : les runs successifs n'utilisent pas tous le même
contrat d'action, la même observation ni la même récompense. Une amélioration de
reward entre deux versions ne prouve donc pas une amélioration du comportement.

Le document technique complémentaire
[REFERENCE_SAC_RESIDUAL_PLACE.md](REFERENCE_SAC_RESIDUAL_PLACE.md) décrit la
configuration **actuelle**. L'ancien [SAC_SYNTHESE.md](../SAC_SYNTHESE.md)
documente le SAC 6D antérieur et est obsolète pour la tâche residual
<code>place</code>.

## Résumé exécutif

Le projet est passé d'un SAC d'assemblage libre à un contrôleur residual tactile
qui suit un chemin géométrique <code>T_ref(s)</code>. SAC commande six
corrections cartésiennes et la progression sur ce chemin. La politique n'a ni
caméra ni pose réelle pièce–gabarit ; elle doit interpréter les contacts au
moyen du wrench, de l'état du contrôleur et d'un historique de huit trames.

Les principaux résultats à ce stade sont les suivants :

- toute la chaîne chemin → simulation → SAC → logs → évaluation fonctionne ;
- les causes de fin, les efforts, les modes de recherche et les composantes de
  récompense sont maintenant traçables dans les CSV ;
- le suivi nominal atteint <code>s=1</code> en 4 s environ, donc l'horizon de
  700 décisions, soit 14 s, n'est pas la cause première des blocages ;
- même sans aucune variabilité et avec une action résiduelle nulle, la vraie
  pose finale de <code>part_1</code> reste décalée d'environ 25,4 mm ;
- le diagnostic géométrique historique détecte une collision sur les trois
  chemins <code>place</code> ;
- le meilleur run évalué avant les derniers correctifs atteint régulièrement le
  voisinage du but, mais termine 10 fois sur 10 au seuil de couple ;
- aucun run évalué n'a encore réussi l'assemblage ;
- le run <code>SAC_12</code>, premier run avec le contrat courant 448D et les
  derniers correctifs, était encore en cours pendant la rédaction. Il ne faut
  pas en tirer de conclusion finale.

Ce n'est pas un dead end : une trajectoire nominale en collision peut servir de
guide à une stratégie tactile. En revanche, le residual RL doit réellement
apprendre à décharger le contact et à corriger l'erreur physique ; « atteindre
la fin du YAML » ne suffit pas.

## 1. Point de départ : ancien SAC d'assemblage libre

Avant cette discussion, le dépôt entraînait un SAC avec une action cartésienne
6D, sans chemin <code>place</code>, avec curriculum et épisodes de
désassemblage. Les artefacts historiques se trouvent directement sous
<code>data/output/</code>, sans sous-dossier par pièce.

Le dernier entraînement de cette génération demandait 500 000 pas et a été
interrompu à 239 685 pas. Son
[rapport d'évaluation](../data/output/evaluation.json) donne :

- 0 succès sur 10 ;
- reward moyen : −110,08 ;
- 570,7 décisions en moyenne ;
- erreurs finales comprises approximativement entre 108 et 236 mm.

Ce résultat constitue une baseline historique uniquement. Il n'est pas
comparable numériquement aux runs residual RL.

## 2. Changement de formulation après la réflexion partagée

La réflexion partagée a conduit à abandonner le suivi d'une trajectoire
temporelle rigide. Le problème a été reformulé comme un chemin géométrique
paramétré par <code>s ∈ [0,1]</code> :

- position interpolée linéairement ;
- orientation interpolée par SLERP ;
- progression, arrêt et recul pilotables ;
- sélection de la pièce avec <code>part_1</code>, <code>part_2</code> ou
  <code>part_3</code> ;
- pièces déjà placées chargées à leur pose finale comme obstacles fixes.

La réalisation correspondante est dans
[load_place_path](../src/place_path.py#L79) et
[PlacePath.pose_at](../src/place_path.py#L54).

Le contrat V1 décidé était :

1. aucune perception visuelle ;
2. une action SAC 7D :
   <code>[vx_res, vy_res, vz_res, wx_res, wy_res, wz_res, progression]</code> ;
3. une admittance cartésienne fixe, non apprise ;
4. des erreurs physiques de gabarit et de prise, constantes pendant un épisode
   et cachées à l'acteur ;
5. une observation tactile/proprioceptive avec historique ;
6. une vraie pose physique réservée au calcul de la récompense et du succès ;
7. des modèles et métriques séparés par pièce.

La construction dynamique du modèle, l'action et l'observation sont regroupées
dans [AssemblyEnv](../src/assembly_env.py#L118).

## 3. Premier smoke test

Le premier lancement court utilisait :

~~~text
ASSEMBLY_PART=part_1
TOTAL_TIMESTEPS=1000
LEARNING_STARTS=100
~~~

Après 700 décisions, un seul épisode était terminé :

- reward : −245,94 ;
- progression : 0,181 ;
- erreur finale : 156,0 mm ;
- force : 1,42 N ;
- succès : faux ;
- fin : limite de temps.

Ce test validait essentiellement la plomberie. Mille transitions, dont les cent
premières avant apprentissage, ne permettaient pas d'évaluer la stratégie.
L'artefact correspondant est
[TensorBoard SAC_1](../data/output/part_1/tensorboard/SAC_1).

## 4. Instrumentation ajoutée pour comprendre les épisodes

La valeur <code>is_success=false</code> ne disait pas pourquoi un épisode
s'arrêtait. Les fins ont donc été séparées en :

- <code>success</code> ;
- <code>unsafe_force</code> ;
- <code>unsafe_torque</code> ;
- <code>unsafe_force_and_torque</code> ;
- <code>recovery_failed</code> ;
- <code>time_limit</code>.

Le CSV Monitor enregistre désormais <code>terminated</code>,
<code>truncated</code>, <code>termination_reason</code>, les erreurs finales,
les progressions finale et maximale, les forces/couples maximaux, les fractions
d'avance/maintien/recul, les modes tactiles, les récupérations, les offsets et
la décomposition de récompense. La liste exacte est
[MONITOR_INFO_KEYWORDS](../src/train.py#L27), et les valeurs sont produites dans
[AssemblyEnv.step](../src/assembly_env.py#L883).

L'évaluation écrit également :

- un JSON détaillé ;
- un CSV aplati, une ligne par épisode ;
- des moyennes et compteurs de déclencheurs.

Voir [_write_episode_csv](../src/evaluate.py#L190) et la
[construction du résumé](../src/evaluate.py#L360).

## 5. Viewer MuJoCo et prétraitement SDF

Le viewer demandé n'est pas Gazebo : la simulation et l'affichage sont MuJoCo.
Le service <code>evaluate</code> a été monté sur X11 et utilise maintenant le
rendu humain par défaut. Le mode sans fenêtre reste disponible avec
<code>EVAL_RENDER=none</code>. Voir le
[service evaluate](../docker-compose.yml#L92).

Les messages « non-manifold » et « mesh is not watertight » viennent du
prétraitement SDF des STL à la compilation du modèle. Le plugin SDF utilisé ne
fournit pas ici de cache disque persistant réutilisable entre conteneurs. Deux
mesures ont néanmoins réduit le coût :

- ne plus appeler <code>mj_setConst</code> à chaque reset ; les transformations
  statiques sont mises à jour directement dans
  [_apply_physical_errors](../src/assembly_env.py#L326) ;
- permettre <code>SKIP_ENV_CHECK=1</code> après une première validation, afin
  d'éviter le deuxième environnement créé par <code>check_env</code>.

Un export SDF exploratoire a aussi été produit dans
[data/output/sdf](../data/output/sdf) : résolution 2 mm, bande 1,5 mm,
760 320 sondes et 91 017 points retenus. Cet export sert au diagnostic ; il
n'est pas un cache chargé par l'environnement d'entraînement.

## 6. Pourquoi conserver 700 décisions

Les premiers logs terminaient très souvent en <code>time_limit</code>. Le calcul
temporel a montré :

- timestep MuJoCo : 1 ms ;
- 20 sous-pas par décision ;
- décision SAC : 20 ms, soit 50 Hz ;
- 700 décisions : 14 s ;
- vitesse nominale <code>ds/dt=0,25</code> ;
- chemin nominal complet : environ 4 s, soit 200 décisions.

Le robot disposait donc d'environ 10 s pour chercher après l'arrivée nominale.
Le blocage était stratégique, pas un simple manque d'horizon. La décision a été
de garder 700 et de n'augmenter cette valeur que si les métriques montrent une
récupération effective interrompue avant convergence.

## 7. Ajout de la récupération près du but

Le recul était auparavant peu attractif et aucun mode explicite ne permettait
de se dégager. Une machine d'état a été ajoutée, d'abord avec
<code>tracking</code>/<code>recovery</code>, puis avec un mode intermédiaire
<code>contact_search</code>.

La logique finale est dans
[_update_recovery_state](../src/assembly_env.py#L565) :

- un contact réel active et mémorise <code>contact_search</code> ;
- 20 N ou 4,5 Nm pendant cinq décisions avec contact déclenchent
  <code>recovery</code> ;
- une stagnation près du but peut également déclencher la récupération ;
- l'avance sur <code>s</code> est interdite en récupération ;
- la sortie demande au moins 0,2 s de récupération puis dix décisions sans
  contact, sous 15 N et 3,5 Nm ;
- une tentative de récupération échoue après 2 s.

La vraie pose relative pièce–gabarit a été ajoutée au MJCF dynamique avec les
capteurs <code>part_rel_pos</code> et <code>part_rel_quat</code>. Elle est lue
par [_true_final_errors](../src/assembly_env.py#L308), mais reste absente de
l'observation SAC.

Des tests couvrent notamment l'entrée par effort, la stagnation, l'interdiction
d'avancer en récupération, la sortie après dégagement et l'impossibilité de
déclarer un faux succès. Voir
[test_recovery_logic.py](../src/test_recovery_logic.py).

## 8. Itérations successives avant la baseline nominale

Le tableau suivant reprend les scalaires finaux disponibles dans TensorBoard.
<code>R</code> et <code>len</code> sont les moyennes glissantes SB3 ; la
progression et l'erreur sont celles du dernier épisode journalisé. Tous les
runs ont un taux de succès enregistré nul.

| Run | Date | Dernier pas | R final | len final | s dernier | Erreur dernière |
|---|---|---:|---:|---:|---:|---:|
| SAC_1 | 06/08 | 700 | −245,9 | 700,0 | 0,181 | 156,0 mm |
| SAC_2 | 06/08 | 149 474 | +86,2 | 674,9 | 0,872 | 22,3 mm |
| SAC_3 | 06/08 | 22 852 | −272,5 | 634,8 | 1,000 | 21,7 mm |
| SAC_4 | 06/08 | 149 531 | −244,6 | 571,0 | 0,564 | 93,5 mm |
| SAC_5 | 06/08 | 24 141 | −310,1 | 524,8 | 0,571 | 87,2 mm |
| SAC_6 | 06/08 | 18 986 | −119,0 | 117,3 | 0,009 | 183,7 mm |
| SAC_7 | 07/08 | 149 814 | −159,7 | 316,1 | 0,872 | 25,5 mm |
| SAC_8 | 07/08 | 41 160 | −314,8 | 686,0 | 0,753 | 41,9 mm |
| SAC_9 | 07/08 | 25 438 | −124,6 | 368,7 | 0,860 | 26,7 mm |
| SAC_10 | 07/08 | 99 904 | −72,7 | 491,5 | 0,860 | 14,0 mm |
| SAC_11 | 07/08 | 42 928 | −135,6 | 660,4 | 0,166 | 180,3 mm |
| SAC_12 | 07/08 | en cours | — | — | — | — |

En particulier, le reward positif de <code>SAC_2</code> malgré zéro succès a
invalidé l'idée qu'un retour croissant suffisait à prouver l'apprentissage de
l'assemblage. Les versions suivantes ont progressivement renforcé le lien entre
reward, vraie pose, sécurité et cause terminale.

Lecture synthétique des essais intermédiaires :

- <code>SAC_1</code> était uniquement le smoke test de 1 000 pas ;
- <code>SAC_2</code> atteignait souvent la zone terminale, mais exploitait une
  récompense encore mal alignée avec le succès ;
- <code>SAC_3</code> a montré qu'atteindre <code>s=1</code> avec environ
  21,7 mm d'erreur et une forte erreur angulaire ne suffisait pas ;
- <code>SAC_4</code> et <code>SAC_5</code> ont régressé en progression et en
  erreur malgré des apprentissages plus longs ou relancés ;
- <code>SAC_6</code> terminait très tôt, ce qui a révélé une logique
  effort/couple trop agressive ;
- <code>SAC_7</code>, après la première récupération explicite, retrouvait
  <code>s≈0,87</code> mais finissait encore en sécurité ;
- <code>SAC_8</code> produisait à nouveau des épisodes proches de 700 pas, d'où
  l'analyse spécifique des time limits ;
- <code>SAC_9</code>, après ajout de <code>contact_search</code>, atteignait
  mieux la zone terminale mais ne réalisait toujours aucun assemblage.

Les itérations intermédiaires ont aussi corrigé :

- l'exclusivité entre <code>terminated</code> et <code>truncated</code> ;
- le timeout par tentative de récupération plutôt que sa durée cumulée ;
- la persistance d'effort et le recul forcé ;
- le mode réellement exécuté dans les métriques ;
- les seuils durs inclusifs ;
- la détection des contacts brefs pendant les 20 sous-pas.

## 9. Essai séparé sur part_3

Le run <code>part_3</code> du 6 août demandait 150 000 pas et a été interrompu à
46 145 pas. Le Monitor contient 92 épisodes terminés :

- 46 limites de temps ;
- 43 seuils de couple ;
- 2 seuils de force ;
- 1 seuil force et couple ;
- 0 succès.

Son [évaluation](../data/output/part_3/evaluation.json) donne :

- 0 succès sur 10 ;
- 10 seuils de couple ;
- reward moyen : −85,61 ;
- 250,8 décisions en moyenne ;
- progression moyenne : 0,669 ;
- erreur position moyenne : 72,1 mm ;
- erreur angulaire moyenne : 12,0° ;
- force maximale moyenne : 16,0 N ;
- couple maximal moyen : 9,07 Nm.

## 10. Variabilité nulle et baseline nominale

Pour isoler la cause, toutes les amplitudes de perturbation de pose ont été
mises à zéro dans [ResidualConfig](../src/assembly_env.py#L45). L'hypothèse
était que le suivi de trajectoire devait alors presque suffire. Ce ne fut pas le
cas.

Un évaluateur nominal indépendant a été ajouté :

- [evaluate_nominal.py](../src/evaluate_nominal.py) ;
- service [evaluate-nominal](../docker-compose.yml#L137) ;
- cible [Makefile](../Makefile#L12).

Il applique une action 7D nulle, désactive recherche/récupération et conserve les
seuils de sécurité. Le
[rapport nominal persistant](../data/output/part_1/nominal_evaluation.json)
contient un épisode :

| Mesure | Résultat |
|---|---:|
| Progression | 1,000 |
| Durée | 700 décisions, 14 s |
| Fin | time_limit |
| Erreur réelle position | 25,423 mm |
| Erreur réelle rotation | 0,041° |
| Force maximale | 54,394 N |
| Couple maximal | 4,072 Nm |
| Durée de contact | 8,18 s |
| Succès | non |

La vraie pose finale présente notamment environ −24,9 mm sur z. Une vérification
console ultérieure avec la source alors courante a retrouvé environ 25,2 mm
d'erreur, mais avec un pic de force différent. La conclusion robuste est que
<code>s=1</code> ne correspond pas à une pièce physiquement assemblée.

## 11. Diagnostic explicite des collisions de chemin

Le test [test_env_colision.py](../src/test_env_colision.py) rejoue les poses
enregistrées par pas de translation d'au plus 2 mm. Le
[rapport du 6 août](../data/output/collision_report.json) trouve une collision
sur les trois segments :

| Pièce | Premier contact | Obstacle(s) | Pénétration maximale |
|---|---:|---|---:|
| part_1 | 1,9046 s | table | 0,814 mm |
| part_2 | 2,1221 s | table | 1,528 mm |
| part_3 | 1,8440 s | table puis part_1 | 0,549 mm |

Ce rapport est historique : il précède le nouveau STL de collision de table et
ne valide donc pas ce nouveau maillage.

L'utilisateur a ensuite fourni
[chandelier_assembly_table_collision.stl](../data/input/cad/chandelier_assembly_table_collision.stl).
Le rendu continue d'utiliser le STL visuel, tandis que les collisions SDF
utilisent le nouveau fichier dans :

- [scene.xml](../data/input/scene.xml#L23) ;
- [_make_model](../src/assembly_env.py#L239) ;
- [test_env_colision.make_model](../src/test_env_colision.py#L75).

## 12. Run SAC_10 : progrès visible, mais échec systématique

<code>SAC_10</code> est arrivé au terme de 100 000 pas. Son
[évaluation déterministe](../data/output/part_1/evaluation.json), associée au
modèle <code>assembly_sac.zip</code>, donne sur dix épisodes :

| Mesure | Résultat |
|---|---:|
| Succès | 0/10 |
| Cause de fin | 10/10 unsafe_torque |
| Reward moyen | −76,045 |
| Décisions moyennes | 374 |
| Progression moyenne | 0,9701 |
| Erreur position moyenne | 32,47 mm |
| Erreur rotation moyenne | 3,98° |
| Force maximale moyenne | 69,0 N |
| Couple maximal moyen | 8,252 Nm |
| Durée de contact moyenne | 3,02 s |
| Impulsion moyenne | 2,153 Ns |
| Récupérations | 0 |

Neuf épisodes sur dix sont presque identiques, avec
<code>s≈0,9673</code>, 31,6 mm d'erreur et un couple maximal d'environ
8,261 Nm. La répétabilité est cohérente avec la variabilité nulle et
l'évaluation déterministe : la politique a appris un comportement reproductible
qui arrive près du but, mais se met en contrainte et dépasse le seuil de couple.

## 13. Analyse de la vidéo

La vidéo [2026-08-07 09-23-42.mkv](../video/2026-08-07%2009-23-42.mkv) dure
environ 55,9 s en 2560×1440 à 60 fps. Les sauts importants visibles toutes les
quelques secondes correspondent aux resets des épisodes.

Les mouvements autour du chemin nominal avant contact n'étaient pas du bruit
d'exploration : [evaluate.py](../src/evaluate.py#L305) appelle la politique avec
<code>deterministic=True</code>. Le diagnostic a identifié plusieurs causes
possibles dans la version alors évaluée :

- les actions résiduelles étaient intégrées dès le début ;
- les états internes de ces intégrateurs étaient insuffisamment observés ;
- le coût portait davantage sur l'action instantanée que sur l'offset accumulé ;
- l'admittance pouvait interpréter une charge inertielle en espace libre comme
  un contact ;
- le wrench n'était pas encore correctement exploité dans le repère monde.

Visuellement, le comportement progressait par rapport aux runs antérieurs, mais
la pièce arrivait décalée, était poussée ou mise en torsion près du but, puis
terminait en sécurité couple.

## 14. Proposition d'un seuil à 80 %, puis retrait

Une rampe d'autorité résiduelle nulle avant environ 80 % du chemin avait été
envisagée pour supprimer les mouvements libres observés sur la vidéo. Cette
proposition était une heuristique, pas une valeur déduite de la géométrie.

L'objection était correcte : si un contact survient avant 80 %, neutraliser SAC
empêcherait précisément l'adaptation recherchée. Cette idée a donc été rejetée
et **n'est pas dans le contrôleur actuel**.

Dans la version finale de cette itération :

- les six actions résiduelles sont actives pour tout <code>s</code>, voir
  [_residual_action_for_mode](../src/assembly_env.py#L523) ;
- un contact peut activer <code>contact_search</code> à n'importe quelle
  progression ;
- <code>s=0,85</code> sert uniquement à la détection de stagnation près du but ;
- l'autorité linéaire élargie de 24 mm dépend du latch tactile ou de la
  récupération, jamais de la seule progression.

## 15. Correctifs finaux après la vidéo

Les changements suivants forment le contrat courant
<code>control_semantics_version=6</code> :

### Observation et contrôle

- ajout de la progression maximale ;
- ajout des modes et du latch tactile ;
- ajout de l'offset résiduel, de l'offset d'admittance et de sa vitesse ;
- passage de 55 à 56 valeurs par trame, donc de 440 à 448 valeurs sur huit
  trames ;
- transformation du wrench du repère poignet vers le monde ;
- admittance active uniquement pendant un contact réel ;
- contact bref mémorisé sur les sous-pas ;
- actions résiduelles toujours disponibles, y compris en recherche et en
  récupération.

### Récompense et terminaisons

- progression récompensée seulement lors d'un nouveau maximum, pour empêcher le
  farming recul/réavance ;
- diminution de la vraie erreur de pose valorisée près du but ;
- coût de l'offset résiduel accumulé ;
- coûts d'effort, corridor, amplitude d'action et changement d'action ;
- reward dense borné à ±0,1 ;
- +250 au succès ;
- −800 à un seuil dur, −900 si force et couple sont simultanément durs ;
- −300 après 2 s de récupération échouée ;
- pénalité de qualité finale bornée à −60 au time limit ;
- <code>gamma=0,999</code> ;
- troncatures à 700 pas traitées comme transitions terminales dans le replay.

Les lignes centrales sont
[AssemblyEnv.step](../src/assembly_env.py#L701) et la
[construction SAC](../src/train.py#L528).

### Tests

La suite contient actuellement 44 tests :

- 3 tests de chargement/interpolation dans
  [test_place_path.py](../src/test_place_path.py) ;
- 41 tests de commande, récupération, observation, récompense et sécurité dans
  [test_recovery_logic.py](../src/test_recovery_logic.py).

Le dernier cycle de modification a été validé dans Docker avec ces 44 tests,
<code>check_env</code> et une intégration MuJoCo/SB3 confirmant une observation
de 448 valeurs.

## 16. Runs après la vidéo

### SAC_11

Le run demandait 100 000 pas et a été interrompu à 42 932 pas. Ses
[métadonnées](../data/output/part_1/training_metadata.json) correspondent encore
à la sémantique v5 : observation 440D, pénalités terminales plus faibles et
activation tactile possible à partir d'un wrench sans contact confirmé.

Il n'a enregistré aucun succès. Sa forte proportion finale de
<code>contact_search</code> a motivé la suppression du faux déclenchement par
charge inertielle. C'est un diagnostic corrélé aux logs, pas une preuve causale
isolée.

### SAC_12

<code>SAC_12</code> a démarré le 7 août vers 11 h 20 avec la sémantique v6,
l'observation 448D et les nouvelles pénalités terminales. Il était encore actif
pendant l'inspection ; le fichier Monitor grandissait pendant sa lecture.

Un instantané vers 11 h 27 montrait environ 21 000 pas, une cinquantaine
d'épisodes et aucun succès. Les causes comprenaient time limits, seuils de
couple/force et récupérations échouées. Ces valeurs ne constituent pas un
résultat final et les rewards ne sont pas comparables à <code>SAC_10</code> :
les pénalités de sécurité sont passées à −800/−900.

## 17. Hypothèses invalidées ou rejetées

| Hypothèse | Verdict |
|---|---|
| « 700 pas sont trop courts » | Invalidée : le nominal finit le chemin en environ 200 pas. |
| « Sans variabilité, le chemin suffit » | Invalidée : 25,4 mm d'erreur réelle à s=1. |
| « Un reward croissant signifie que l'assemblage progresse » | Invalidée : SAC_2 a un reward positif mais zéro succès. |
| « Le wrench élevé signifie forcément contact » | Invalidée : des charges inertielles existent en espace libre. |
| « Il faut couper SAC avant 80 % » | Rejetée : un contact peut survenir avant ce seuil. |
| « La progression à 100 % suffit au succès » | Invalidée : le succès utilise la vraie pose physique. |
| « Le STL visuel convient nécessairement aux collisions » | Invalidée : un mesh collision séparé a été fourni. |
| « Une trajectoire nominale en collision est un dead end » | Rejetée sous condition : le contact doit être récupérable et rester sous les limites de sécurité. |

## 18. État actuel et travail restant

Le socle expérimental est beaucoup plus fiable qu'au début : observation des
états internes, vraie métrique de pose, sécurité sous-pas, causes terminales,
baseline nominale et logs exploitables. En revanche, aucune preuve de succès
appris n'existe encore.

Les points ouverts les plus importants sont :

1. laisser finir puis évaluer <code>SAC_12</code> avec des seeds figées ;
2. vérifier à nouveau les collisions des trois chemins avec le **nouveau** STL
   de table ;
3. comparer nominal, nominal + admittance, récupération déterministe et residual
   SAC sur exactement les mêmes scénarios ;
4. n'activer les perturbations physiques qu'après obtention d'un comportement
   fiable à variabilité nulle ;
5. traiter les limites techniques listées dans le document de référence,
   notamment friction/bruit non randomisés, impulsion non récompensée, absence
   de sauvegarde du replay buffer et absence de meilleur modèle automatique.
