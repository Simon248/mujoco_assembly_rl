# Référence technique — SAC residual tactile sur place

État de la **source courante** au 7 août 2026. Pour la chronologie des essais,
voir [HISTORIQUE_RESIDUAL_RL_PLACE.md](HISTORIQUE_RESIDUAL_RL_PLACE.md).

Cette page décrit ce que le code exécute réellement, y compris les écarts avec
le plan initial. La source fait foi :

- configuration physique et récompense :
  [ResidualConfig](../src/assembly_env.py#L20) ;
- environnement :
  [AssemblyEnv](../src/assembly_env.py#L118) ;
- création de SAC :
  [src/train.py](../src/train.py#L528) ;
- valeurs Docker effectives :
  [docker-compose.yml](../docker-compose.yml#L56).

## 1. Fiche synthétique

| Élément | Valeur courante |
|---|---|
| Tâche | residual RL tactile, segment <code>place</code> uniquement |
| Algorithme | Soft Actor-Critic, politique MLP |
| Pièce | <code>part_1</code>, <code>part_2</code> ou <code>part_3</code> |
| Action | 7 valeurs continues dans [−1, 1] |
| Observation | 56 valeurs par trame × 8 trames = 448 |
| Caméra / pose estimée | aucune |
| Pose réelle pièce–gabarit dans l'observation | non |
| Fréquence de décision | 50 Hz |
| Horizon | 700 décisions = 14 s |
| Temps nominal pour s=1 | environ 4 s |
| Variabilité de pose active | aucune : toutes les amplitudes valent zéro |
| Réseau | MLP [256, 256] |
| Learning rate | 3×10⁻⁴ |
| Replay buffer via Compose | 50 000 transitions |
| Batch via Compose | 256 |
| Début de l'apprentissage | 5 000 transitions |
| Gamma | 0,999 |
| Tau | 0,005 |
| Succès | s≥0,999, erreur vraie <3 mm et <4°, aucun seuil dur |
| Seuils doux | 20 N et 4,5 Nm |
| Seuils durs | 80 N et 8 Nm |

Le flux de contrôle peut se résumer ainsi :

~~~text
YAML segments.place ──> T_ref(s) ──────────────────────────────┐
                                                              │
observation 448D ──> SAC ──> action 7D                        │
                            ├─ progression ──> nouveau s       │
                            └─ vitesses 6D ──> offset résiduel ├─> cible portique
                                                              │
contact + wrench ──> admittance fixe ──> offset compliant ────┘

pose réelle pièce–gabarit ──> reward / succès / métriques seulement
~~~

## 2. Chemin géométrique et sélection de pièce

### Chargement

[load_place_path](../src/place_path.py#L79) recherche exactement un fichier :

~~~python
matches = sorted(
    Path(paths_dir).glob(f"chandelier_{part_name}_place*.yaml")
)
points = document.get("segments", {}).get("place")
~~~

Seul <code>segments.place</code> est utilisé. Les temps enregistrés dans le YAML
ne pilotent pas l'exécution.

La progression est calculée à partir de la longueur d'arc de translation puis
normalisée :

~~~python
distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
distances = np.maximum(distances, 1e-9)
progress = np.concatenate([[0.0], np.cumsum(distances)])
progress /= progress[-1]
~~~

Référence :
[src/place_path.py, lignes 89–97](../src/place_path.py#L89).

[PlacePath.pose_at](../src/place_path.py#L54) interpole la position linéairement
et l'orientation par SLERP sur le chemin quaternionique le plus court.

### Pièce active

La CLI accepte uniquement <code>part_1</code>, <code>part_2</code> et
<code>part_3</code> dans
[train.parse_args](../src/train.py#L204). Compose expose le choix par
<code>ASSEMBLY_PART</code> :

~~~yaml
ASSEMBLY_PART: "${ASSEMBLY_PART:-part_1}"
~~~

Référence : [docker-compose.yml, ligne 11](../docker-compose.yml#L11).

Pour <code>part_2</code> et <code>part_3</code>, les pièces précédentes sont
créées comme obstacles fixes à leur pose finale. Elles suivent l'erreur physique
du gabarit. Voir [_make_model](../src/assembly_env.py#L228), en particulier les
[lignes 247–253](../src/assembly_env.py#L247).

### Point d'attention sur scene.xml

Le chemin <code>scene.xml</code> est vérifié et son dossier sert à déduire
<code>cad/</code> et <code>chemin/</code>, mais l'environnement ne charge pas
directement tout son contenu. [_make_model](../src/assembly_env.py#L228)
reconstruit et compile un MJCF temporaire. Une modification de
<code>scene.xml</code> doit donc être reproduite dans cette fonction si elle
concerne le modèle effectivement entraîné.

## 3. Temps de simulation et horizon

Les valeurs sont réparties entre
[AssemblyEnv.__init__](../src/assembly_env.py#L127) et
[_make_model](../src/assembly_env.py#L228) :

~~~python
max_episode_steps = 700
frame_skip = 20
decision_dt = 0.02
~~~

~~~xml
<option timestep="0.001" .../>
~~~

Il en résulte :

- 1 sous-pas MuJoCo = 1 ms ;
- 20 sous-pas par décision SAC ;
- 1 décision = 20 ms ;
- 50 décisions par seconde ;
- 700 décisions = 14 s simulées.

Avec <code>progress_speed=0,25 s⁻¹</code>, une action de progression nulle en
<code>tracking</code> parcourt le chemin en 200 décisions, soit environ 4 s.

Il n'existe pas d'assertion vérifiant que
<code>decision_dt == frame_skip × model.opt.timestep</code>. Si l'un de ces
paramètres est modifié, il faut maintenir cette cohérence manuellement.

## 4. Action SAC

L'espace d'action est défini dans
[AssemblyEnv.__init__](../src/assembly_env.py#L167) :

~~~python
self.action_space = spaces.Box(
    -1.0, 1.0, shape=(7,), dtype=np.float32
)
~~~

| Indice | Commande | Échelle maximale |
|---:|---|---:|
| 0 | vitesse d'offset X | ±20 mm/s |
| 1 | vitesse d'offset Y | ±20 mm/s |
| 2 | vitesse d'offset Z | ±20 mm/s |
| 3 | vitesse résiduelle roll | ±20°/s |
| 4 | vitesse résiduelle pitch | ±20°/s |
| 5 | vitesse résiduelle yaw | ±20°/s |
| 6 | commande de progression | sémantique dépendant du mode |

Les échelles viennent de
[ResidualConfig](../src/assembly_env.py#L24) :

~~~python
residual_linear_speed = 0.020
residual_angular_speed = np.deg2rad(20.0)
progress_speed = 0.25
decision_dt = 0.02
~~~

### Intégration des six résiduels

Les six premières valeurs ne sont pas des offsets instantanés : ce sont des
vitesses intégrées dans un état persistant.

~~~python
velocity_scale = np.array(
    [c.residual_linear_speed] * 3
    + [c.residual_angular_speed] * 3
)
self._residual_offset += (
    residual_action * velocity_scale * c.decision_dt
)
~~~

Référence :
[AssemblyEnv.step, lignes 723–728](../src/assembly_env.py#L723).

À amplitude 1 pendant une décision :

- un axe linéaire change de 0,4 mm ;
- un axe angulaire change de 0,4°.

L'offset est ensuite borné **par composante**, pas par norme :

- translation : ±15 mm avant latch tactile ;
- translation : ±24 mm après latch, en recherche ou récupération ;
- rotation : ±12° dans tous les modes.

Les six résiduels restent actifs dès <code>s=0</code> et dans tous les modes :

~~~python
def _residual_action_for_mode(action, mode):
    del mode
    return np.asarray(action[:6], dtype=np.float64).copy()
~~~

Référence :
[_residual_action_for_mode](../src/assembly_env.py#L523).
Il n'existe aucun seuil d'activation à 80 %.

### Commande de progression

La fonction exacte est
[_effective_progress_request](../src/assembly_env.py#L479).

#### tracking

~~~python
if requested_progress <= 0.0:
    return 1.0 + requested_progress
return 1.0 + 0.5 * requested_progress
~~~

| Action 7 | Requête effective | ds/dt | Effet |
|---:|---:|---:|---|
| −1 | 0 | 0 | arrêt |
| 0 | 1 | +0,25 s⁻¹ | vitesse nominale |
| +1 | 1,5 | +0,375 s⁻¹ | avance accélérée |

Le mode <code>tracking</code> permet d'avancer ou de s'arrêter, mais **pas de
reculer**.

#### contact_search

~~~python
np.clip(
    contact_search_nominal_request + requested_progress,
    -1.0,
    contact_search_max_forward_request,
)
~~~

Avec les valeurs 0,25 et 0,50 :

| Action 7 | Requête effective | ds/dt | Effet |
|---:|---:|---:|---|
| −1 | −0,75 | −0,1875 s⁻¹ | recul |
| −0,25 | 0 | 0 | maintien |
| 0 | +0,25 | +0,0625 s⁻¹ | avance lente |
| +1 | +0,50 | +0,125 s⁻¹ | avance maximale |

Une requête positive est encore multipliée par
[_contact_progress_scale](../src/assembly_env.py#L508), qui décroît de 1 à 0
entre les seuils doux et durs d'effort.

#### recovery

~~~python
return (
    -1.0
    if self._forced_retreat
    else min(requested_progress, 0.0)
)
~~~

SAC peut maintenir ou reculer. Une action positive ne peut jamais augmenter
<code>s</code>. Si le recul est forcé, la requête vaut −1 indépendamment de
l'action 7.

## 5. Observation

### Taille et historique

La taille est définie dans
[AssemblyEnv.__init__](../src/assembly_env.py#L167) :

~~~python
self._base_observation_size = 56
history_length = 8
self.observation_space = spaces.Box(
    -np.inf,
    np.inf,
    shape=(56 * history_length,),
    dtype=np.float32,
)
~~~

Une trame contient 56 valeurs et huit trames sont concaténées, soit
**448 valeurs**. [_get_obs](../src/assembly_env.py#L431) ordonne l'historique de
la plus ancienne à la plus récente. Au reset, les huit trames sont des copies de
la première observation ; elles sont ensuite remplacées une par une.

Il ne s'agit pas d'une politique récurrente. Le MLP reçoit un vecteur aplati.

### Carte exacte d'une trame

La composition est dans [_base_obs](../src/assembly_env.py#L416).

| Indices | Taille | Contenu | Normalisation |
|---:|---:|---|---|
| 0–2 | 3 | position portique − point courant du chemin | division par 20 mm, clip ±5 |
| 3–5 | 3 | angles portique − angles courants du chemin | division par 0,25 rad, clip ±5 |
| 6–8 | 3 | position portique − pose finale nominale | division par 0,15 / 0,15 / 0,25 m, clip ±5 |
| 9–11 | 3 | angles portique − angles finaux nominaux | division par 0,7 rad, clip ±5 |
| 12 | 1 | progression courante s | brute |
| 13 | 1 | progression maximale atteinte | brute |
| 14 | 1 | vitesse de progression ds/dt | brute |
| 15–17 | 3 | vitesse linéaire du portique | division par 0,10 m/s |
| 18–20 | 3 | vitesse angulaire du portique | division par 1 rad/s |
| 21–23 | 3 | force X/Y/Z dans le monde | division par 50 N |
| 24–26 | 3 | couple X/Y/Z dans le monde | division par 5 Nm |
| 27–33 | 7 | action précédente | déjà dans [−1,1] |
| 34 | 1 | contact courant | booléen |
| 35 | 1 | mode contact_search | booléen |
| 36 | 1 | mode recovery | booléen |
| 37–42 | 6 | offset résiduel intégré | division par ses limites |
| 43–48 | 6 | offset d'admittance | division par 0,006 |
| 49–54 | 6 | vitesse d'admittance | division par 0,040 |
| 55 | 1 | latch tactile contact_search | booléen |

La trame historique <code>k</code> occupe les indices
<code>56k ... 56k+55</code>.

Les erreurs d'orientation nominales sont des différences d'angles Euler XYZ,
pas une erreur quaternionique. Cette représentation repose sur les petites
rotations et les domaines bornés du portique.

### Données volontairement cachées

La vraie pose relative pièce–gabarit est calculée par
[_true_final_errors](../src/assembly_env.py#L308) :

~~~python
relative_pos = self._sensor(self.true_rel_pos_sensor_id)
relative_quat = self._sensor(self.true_rel_quat_sensor_id)
position_error = np.linalg.norm(relative_pos - target_pos)
rotation_error = _quat_angle(
    _quat_multiply(_quat_inverse(target_quat), relative_quat)
)
~~~

Elle sert à la récompense, à la stagnation, au succès et aux métriques, mais
n'est pas concaténée dans <code>_base_obs</code>. Les erreurs physiques de
gabarit et de prise sont également cachées.

Il n'y a :

- ni caméra ;
- ni estimation de pose ;
- ni état privilégié fourni au critic ;
- ni critic asymétrique.

Le problème vu par SAC est donc partiellement observable. L'historique de huit
trames, soit 160 ms, est l'approximation utilisée en V1.

## 6. Contacts, wrench et admittance

### Contact géométrique

[_has_contact](../src/assembly_env.py#L341) cherche un contact entre la géométrie
de la pièce active et :

- la table ;
- les pièces déjà assemblées.

La détection est répétée à chaque sous-pas dans
[_run_control_substeps](../src/assembly_env.py#L675). Un contact bref peut donc
être mémorisé même s'il a disparu à la fin des 20 ms.

### Wrench

[_wrench](../src/assembly_env.py#L293) :

1. lit force et couple au site poignet ;
2. soustrait le biais mesuré au reset ;
3. transforme force et couple du repère du site vers le repère monde.

~~~python
site_to_world = self.data.site_xmat[
    self.wrist_site_id
].reshape(3, 3)
world_force = site_to_world @ local_wrench[:3]
world_torque = site_to_world @ local_wrench[3:]
~~~

### Loi d'admittance

La loi fixe est dans
[_update_admittance](../src/assembly_env.py#L348) :

~~~python
acceleration = (
    sign * wrench
    - damping * velocity
    - stiffness * offset
) / mass
~~~

| Paramètre | Valeur |
|---|---:|
| Masse virtuelle | 3 |
| Amortissement | 90 |
| Raideur | 900 |
| Signe du wrench | −1 |
| Offset maximal par composante | ±0,006 m ou ±0,006 rad |
| Vitesse maximale par composante | ±0,040 m/s ou rad/s |
| Accélération maximale par composante | ±2 m/s² ou rad/s² |

Le signe −1 déplace la cible dans le sens qui décharge le wrench mesuré.

L'admittance reçoit un wrench non nul uniquement si un contact est présent au
début de la décision :

~~~python
self._update_admittance(
    wrench_before,
    tactile_active=self._has_contact(),
)
~~~

Référence :
[AssemblyEnv.step, lignes 729–732](../src/assembly_env.py#L729).
En espace libre, raideur et amortissement ramènent l'offset vers zéro. Si un
contact apparaît pendant les sous-pas, l'admittance y réagit à la décision
suivante ; la sécurité, elle, voit déjà le pic dans la décision courante.

## 7. Modes tactiles et récupération

### tracking

Mode initial après chaque [reset](../src/assembly_env.py#L930). SAC commande les
sept actions et suit normalement le chemin.

### contact_search

[_enter_contact_search](../src/assembly_env.py#L439) est appelé lorsqu'un contact
géométrique est détecté, quelle que soit la valeur de <code>s</code>. Le latch
reste vrai jusqu'au prochain reset. Une perte de contact ne ramène donc pas le
contrôleur en <code>tracking</code>.

Ce latch :

- ralentit la progression ;
- autorise le recul ;
- élargit la limite résiduelle linéaire de 15 à 24 mm ;
- reste observable par SAC.

Un wrench inertiel sans collision ne peut plus l'activer.

### recovery : entrée

La logique est dans
[_update_recovery_state](../src/assembly_env.py#L565).

Une récupération est déclenchée si :

1. force ≥20 N pendant 5 décisions consécutives avec contexte de contact ;
2. couple ≥4,5 Nm pendant 5 décisions consécutives avec contexte de contact ;
3. stagnation de pose près du but.

La stagnation conserve 51 échantillons, représentant 50 intervalles, donc
environ 1 s :

- à partir de <code>s≥0,85</code> si une avance effective >0,1 est demandée ;
- toujours à partir de <code>s≥0,98</code> si la pose finale reste hors
  tolérance ;
- déclenchement si le gain de position est <0,5 mm **et** le gain de rotation
  est <0,5°.

Une dégradation de l'une de ces erreurs est donc également considérée comme une
amélioration insuffisante.

### recovery : commande, sortie et échec

Pendant la récupération :

- les six résiduels restent libres ;
- l'avance sur <code>s</code> est impossible ;
- un effort doux encore élevé force la requête de progression à −1.

La sortie exige simultanément :

- au moins 10 décisions de récupération, soit 0,2 s ;
- 10 décisions consécutives sans contact ;
- force <15 N ;
- couple <3,5 Nm.

Après la sortie, le latch étant conservé, le mode redevient normalement
<code>contact_search</code>. Une tentative qui atteint 2 s se termine par
<code>recovery_failed</code>. Voir
[_recovery_has_failed](../src/assembly_env.py#L669).

Point connu : <code>recovery_effort_persistence_steps=25</code> ne crée pas
actuellement une vraie persistance après disparition de l'effort, car le
compteur est remis à zéro dès que l'effort redescend.

## 8. Cible finale et bornes mécaniques

La cible de position du portique est :

~~~python
target = (
    self._path_qpos(self._progress)
    + self._residual_offset
    + self._admittance_offset
)
self.data.ctrl[self.actuator_ids] = np.clip(
    target,
    ctrl_range[:, 0],
    ctrl_range[:, 1],
)
~~~

Référence :
[AssemblyEnv.step, lignes 733–735](../src/assembly_env.py#L733).

| Borne | Avant latch tactile | Après latch / recovery |
|---|---:|---:|
| Offset résiduel linéaire, par axe | 15 mm | 24 mm |
| Offset résiduel angulaire, par axe | 12° | 12° |
| Corridor translation | 20 mm | 32 mm |
| Offset admittance, par composante | 6 mm / 0,006 rad | identique |

Les courses du portique définies dans
[_make_model](../src/assembly_env.py#L245) sont :

- X et Y : [−0,15 ; +0,15] m ;
- Z : [−0,24 ; +0,05] m autour de l'offset de base ;
- roll, pitch, yaw : [−0,7 ; +0,7] rad.

Les limites d'effort des actionneurs sont définies par
[ResidualConfig.actuator_force_limits](../src/assembly_env.py#L37) :

- X : 250 N ;
- Y : 250 N ;
- Z : 300 N ;
- roll, pitch, yaw : 30 Nm.

Il n'existe pas de clip explicite final sur <code>qvel</code>. Les vitesses
résiduelles et d'admittance sont bornées, puis la dynamique des actionneurs de
position produit le mouvement.

## 9. Sécurité

Les seuils dans [ResidualConfig](../src/assembly_env.py#L31) sont :

| Type | Force | Couple |
|---|---:|---:|
| Doux | 20 N | 4,5 Nm |
| Dur | 80 N | 8 Nm |
| Sortie de recovery | <15 N | <3,5 Nm |

Les comparaisons dures sont inclusives :

~~~python
return (
    force >= self.config.hard_force,
    torque >= self.config.hard_torque,
)
~~~

Référence : [_is_hard_unsafe](../src/assembly_env.py#L412).

Les pics sont vérifiés après chaque sous-pas de 1 ms. La décision MuJoCo est
interrompue dès qu'un seuil dur est atteint. Contrairement au plan initial, le
code courant n'exécute pas ensuite un mouvement physique de recul d'urgence :
il arrête l'épisode et applique la pénalité terminale.

L'impulsion enregistrée est une mesure conservative :

~~~python
self._impulse += max(
    0.0, peak_force - c.soft_force
) * c.decision_dt
~~~

Elle utilise le pic de décision, pas une intégrale exacte des 20 sous-pas.

## 10. Récompense réellement optimisée

SAC maximise l'espérance du retour actualisé, auquel son objectif d'entropie
ajoute l'exploration. Le reward de l'environnement est construit dans
[AssemblyEnv.step](../src/assembly_env.py#L701).

### Progression nouvelle

Seule une portion de chemin jamais atteinte auparavant rapporte :

~~~python
new_progress = max(
    float(progress) - self._max_progress,
    0.0,
)
self._max_progress = max(
    self._max_progress,
    float(progress),
)
~~~

Référence :
[_update_progress_frontier](../src/assembly_env.py#L502).

Le poids vaut 15 jusqu'à <code>s=0,85</code>, puis décroît linéairement jusqu'à
zéro à <code>s=0,98</code>. Reculer n'a pas de pénalité directe, mais réavancer
sur une portion déjà visitée ne rapporte rien.

### Amélioration de la vraie pose

Près du but, <code>pose_focus</code> monte linéairement de 0 à 1 entre
<code>s=0,85</code> et <code>s=0,98</code> :

~~~python
task_reward += pose_focus * (
    500.0 * position_improvement
    + 20.0 * rotation_improvement
)
~~~

Les améliorations sont les différences de vraie erreur physique entre deux
décisions. Elles sont signées : améliorer rapporte, empirer pénalise.

Il n'y a pas de pénalité dense absolue de l'erreur angulaire à chaque pas.
L'angle intervient par son amélioration, le critère de succès et la pénalité de
qualité au time limit.

### Coût d'effort

Force et couple ne coûtent rien sous les seuils doux. Au-dessus, leurs excès
normalisés et saturés sont mis au carré :

~~~python
effort_cost = effort_weight * (
    force_excess**2 + torque_excess**2
)
~~~

Le poids vaut :

- 0,5 en <code>tracking</code> ;
- 0,25 en <code>contact_search</code> et <code>recovery</code>.

### Corridor, action et variation

~~~python
corridor_cost = 0.1 * normalized_excess**2
action_cost = 0.002 * np.dot(action, action)
action_change_cost = 0.005 * np.dot(
    action - previous_action,
    action - previous_action,
)
~~~

Le corridor ne considère que la translation. Le coût d'action porte sur les
sept composantes, progression incluse.

### Offset accumulé

[_residual_offset_cost](../src/assembly_env.py#L403) évite qu'une faible action
maintenue longtemps construise gratuitement un grand décalage :

~~~python
0.01 * ||offset_linear_normalisé||²
+ 0.005 * ||offset_angulaire_normalisé||²
~~~

### Bonus de déchargement

En récupération seulement, une baisse de force peut apporter jusqu'à +0,05 :

~~~python
0.05 * clip(
    (previous_force - force) / soft_force,
    0.0,
    1.0,
)
~~~

Rester en contact ne procure aucun bonus.

### Clipping dense

Le reward de tâche est d'abord borné à [−0,1 ; +0,1]. Les coûts sont soustraits,
puis le dense final est à nouveau borné à cet intervalle :

~~~python
dense_reward = np.clip(
    task_reward
    - effort_cost
    - corridor_cost
    - action_cost
    - action_change_cost
    - offset_cost,
    -0.1,
    0.1,
)
~~~

### Récompenses terminales

| Cause terminale | Valeur ajoutée |
|---|---:|
| Succès | +250 |
| Force **ou** couple dur | −800 |
| Force **et** couple durs | −900 |
| Recovery échouée | −300 |
| Time limit | de 0 à −60 selon la pose finale |

La pénalité time limit comprend :

- jusqu'à −40 entre 3 et 30 mm d'erreur position ;
- jusqu'à −20 entre 4 et 20° d'erreur rotation.

Le reward du dernier pas vaut <code>dense_reward + terminal_reward</code>.

### Mesures qui ne sont pas dans la récompense

Les métriques suivantes sont journalisées mais ne sont pas directement
optimisées :

- durée de contact ;
- impulsion de contact ;
- nombre de récupérations ;
- durée de récupération ;
- force/couple maximaux en dessous des terminaisons, hors coût dense d'excès.

## 11. Succès et causes de fin

Le succès est calculé à la
[ligne 815 d'AssemblyEnv.step](../src/assembly_env.py#L815) :

~~~python
success = (
    self._progress >= 0.999
    and final_error < 0.003
    and final_rot_error < np.deg2rad(4.0)
    and not unsafe
)
~~~

La pose est la vraie pose physique, pas la commande nominale.

| Cause | terminated | truncated |
|---|---:|---:|
| success | vrai | faux |
| unsafe_force | vrai | faux |
| unsafe_torque | vrai | faux |
| unsafe_force_and_torque | vrai | faux |
| recovery_failed | vrai | faux |
| time_limit à 700 pas | faux | vrai |
| épisode en cours | faux | faux |

Le calcul exclusif est dans
[AssemblyEnv.step, lignes 828–832](../src/assembly_env.py#L828). Les champs et
la raison textuelle sont inclus dans <code>info</code>.

## 12. Paramètres d'apprentissage SAC

La construction explicite est dans [train.main](../src/train.py#L528) :

~~~python
model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    buffer_size=buffer_size,
    learning_starts=args.learning_starts,
    batch_size=batch_size,
    tau=0.005,
    gamma=0.999,
    replay_buffer_kwargs={
        "handle_timeout_termination": False,
    },
    train_freq=1,
    gradient_steps=1,
    policy_kwargs={"net_arch": [256, 256]},
    seed=args.seed,
    device=device,
)
~~~

### Valeurs effectives avec Docker Compose

| Paramètre | Valeur |
|---|---:|
| Algorithme | SAC |
| Politique | MlpPolicy |
| Couches cachées | [256, 256] |
| Learning rate | 3×10⁻⁴ constant |
| Buffer | 50 000 |
| Batch | 256 |
| learning_starts | 5 000 |
| Gamma | 0,999 |
| Tau Polyak | 0,005 |
| train_freq | 1 décision |
| gradient_steps | 1 par décision après warm-up |
| Environnements | 1 |
| Seed | 7 par défaut |
| Device | CUDA par défaut |
| Pas demandés | 500 000 par défaut |
| Checkpoint | tous les 50 000 pas |
| Log console | tous les 250 pas |

Les valeurs Compose sont aux
[lignes 69–89](../docker-compose.yml#L69). En lançant
<code>python -m src.train</code> directement hors Compose, les fallbacks de
[train.py](../src/train.py#L516) sont :

- device CPU ;
- buffer 10 000 ;
- batch 64.

### Warm-up et exploration

Pendant les 5 000 premiers pas, SB3 remplit le replay avec des actions
aléatoires. Ces actions concernent les sept dimensions et peuvent donc créer des
mouvements résiduels en espace libre. Ensuite, l'acteur stochastique SAC collecte
les transitions pendant l'entraînement.

En évaluation, l'action est déterministe :

~~~python
action, _ = model.predict(
    obs,
    deterministic=True,
)
~~~

Référence : [evaluate.py, lignes 305–308](../src/evaluate.py#L305).

### Paramètres SB3 laissés aux valeurs par défaut

La version installée lors du checkpoint courant est Stable-Baselines3 2.9.0.
Avec cette version, les paramètres non fournis par le dépôt impliquent
notamment :

- deux critics Q ;
- activation ReLU ;
- optimiseur Adam ;
- coefficient d'entropie appris automatiquement ;
- cible d'entropie automatique, soit −7 pour sept actions ;
- replay uniforme, sans priorité ;
- transitions à un pas ;
- pas de bruit d'action externe ;
- pas de gSDE.

Ces éléments ne sont pas tous figés dans le dépôt :
[requirements.txt](../requirements.txt#L1) ne versionne ni
<code>stable-baselines3</code> ni <code>gymnasium</code>. Une reconstruction
future peut donc changer leurs défauts.

### Gamma et time limits

À <code>gamma=0,999</code> :

- <code>gamma^700 ≈ 0,496</code> ;
- la demi-vie est d'environ 693 décisions ;
- à 50 Hz, cela représente environ 13,9 s.

Une conséquence terminale à la fin d'un épisode de 700 pas conserve donc
environ la moitié de son poids vu depuis le début.

~~~python
replay_buffer_kwargs={
    "handle_timeout_termination": False,
}
~~~

Référence : [train.py, lignes 537–539](../src/train.py#L537).
Les troncatures à 700 décisions sont apprises comme de vraies fins d'épisode :
le critic ne bootstrappe pas au-delà du reset.

### Ce qui est optimisé

Après le warm-up, chaque décision ajoute une transition au replay puis produit
une mise à jour par gradient sur un batch de 256. SAC apprend :

- deux fonctions Q du retour actualisé ;
- une politique stochastique qui maximise Q tout en conservant de l'entropie ;
- automatiquement le coefficient d'entropie.

L'acteur et les critics reçoivent tous la même observation 448D. Le critic ne
voit pas la pose réelle cachée, même si cette pose détermine une partie du
reward.

## 13. Variabilité physique

### Mécanismes disponibles

[_sample_error](../src/assembly_env.py#L323) tire indépendamment chaque
composante selon une loi normale centrée :

~~~python
np_random.normal(0.0, linear_sigma, 3)
np_random.normal(0.0, angular_sigma, 3)
~~~

Trois familles existent :

| Famille | Effet |
|---|---|
| initial | erreur du portique autour du premier point du chemin |
| fixture | translation/rotation physique du gabarit |
| grasp | translation/rotation physique de la pièce par rapport au poignet |

Les sigmas angulaires sont en radians. Les Gaussiennes ne sont pas tronquées.
Les erreurs de fixture et de grasp restent constantes pendant tout l'épisode et
ne sont pas observées par SAC.

### Valeurs réellement actives

Les six amplitudes courantes valent zéro dans
[ResidualConfig](../src/assembly_env.py#L45) :

~~~python
initial_linear_error = 0.0
initial_angular_error = 0.0
fixture_linear_error = 0.0
fixture_angular_error = 0.0
grasp_linear_error = 0.0
grasp_angular_error = 0.0
~~~

[train.main](../src/train.py#L484) construit <code>AssemblyEnv</code> sans lui
passer de configuration personnalisée. **L'entraînement courant n'a donc aucune
variabilité aléatoire de pose.**

Pour les réactiver, il faut actuellement modifier <code>ResidualConfig</code>
ou construire l'environnement par code avec une autre instance de configuration.
Aucune variable Compose ne les surcharge.

La stochasticité qui subsiste pendant l'entraînement vient des actions
aléatoires du warm-up, puis de l'échantillonnage de la politique SAC. Avec les
six sigmas à zéro, changer le seed d'un épisode d'évaluation déterministe ne
change pas sa géométrie physique ; cela explique la forte répétabilité de
certaines évaluations.

### Séquence de reset

[AssemblyEnv.reset](../src/assembly_env.py#L930) :

1. tire les erreurs fixture et grasp ;
2. réinitialise les données MuJoCo ;
3. déplace physiquement gabarit, prise et pièces fixes ;
4. remet <code>s</code>, offsets, historique et modes à zéro ;
5. tire l'erreur initiale du portique ;
6. effectue 30 sous-pas de stabilisation ;
7. prend la mesure F/T obtenue comme biais de tare ;
8. initialise l'observation et les erreurs vraies précédentes.

Les transformations statiques sont appliquées par
[_apply_physical_errors](../src/assembly_env.py#L326) sans
<code>mj_setConst</code>, afin de ne pas réinitialiser les plugins SDF à chaque
épisode.

### Variabilités prévues mais absentes

Malgré le plan initial, la source courante n'implémente pas :

- de randomisation de friction ;
- de bruit F/T injecté ;
- de biais F/T aléatoire ;
- de variation de masse ou de dynamique.

La friction est fixe :

~~~xml
<geom friction="0.9 0.01 0.001" .../>
~~~

Référence : [_make_model, ligne 234](../src/assembly_env.py#L234).
Le « biais » F/T courant est seulement la tare mesurée après stabilisation, pas
un tirage aléatoire. Les perturbations tirées ne sont pas enregistrées dans le
CSV, ce qui empêcherait pour l'instant une analyse scénario par scénario si
elles étaient réactivées.

## 14. Géométrie et collisions

Le modèle courant distingue :

- [chandelier_assembly_table_visual.stl](../data/input/cad/chandelier_assembly_table_visual.stl)
  pour le rendu ;
- [chandelier_assembly_table_collision.stl](../data/input/cad/chandelier_assembly_table_collision.stl)
  pour la collision SDF.

La sélection effective est visible dans
[_make_model, lignes 239–244](../src/assembly_env.py#L239).

MuJoCo compile le STL avec le plugin
<code>mujoco.sdf.sdflib</code>. Les avertissements non-manifold/watertight sont
donc attendus au démarrage tant que les meshes ont ces défauts. Il n'y a pas de
cache SDF persistant chargé depuis <code>data/output</code>. Le modèle n'est en
revanche plus recompilé à chaque reset.

Le rapport historique de collisions
[collision_report.json](../data/output/collision_report.json) a été produit
avant l'ajout du nouveau mesh de table. Il doit être régénéré avant de conclure
sur la géométrie actuelle.

## 15. Entraînement, sauvegardes et reprise

Les sorties sont séparées par pièce dans
[train.main](../src/train.py#L397) :

~~~text
data/output/<part>/
├── models/
│   ├── assembly_sac.zip
│   └── assembly_sac_interrupted.zip
├── checkpoints/
├── monitor/train.monitor.csv
├── tensorboard/
└── training_metadata.json
~~~

Les checkpoints sont configurés dans
[CheckpointCallback](../src/train.py#L492) :

~~~python
save_replay_buffer=False
save_vecnormalize=False
~~~

Conséquences :

- un checkpoint sauvegarde les réseaux, pas le replay buffer ;
- relancer <code>train</code> crée un nouveau SAC au lieu de reprendre le
  précédent ;
- une interruption clavier sauvegarde la politique partielle et écrit des
  métadonnées, voir [train.py, lignes 592–614](../src/train.py#L592) ;
- une reprise exacte de l'optimisation n'est pas disponible.

Le même nom de Monitor et les mêmes noms de checkpoints sont réutilisés lors
des runs suivants d'une pièce. TensorBoard crée des sous-runs numérotés, mais
les CSV/checkpoints/modèles peuvent être remplacés. Il faut archiver un run
avant d'en lancer un autre si sa traçabilité complète est importante.

Les métadonnées ne sont écrites qu'à la fin normale ou lors d'un
<code>KeyboardInterrupt</code>. Pendant un run actif, le fichier
<code>training_metadata.json</code> peut donc encore décrire le run précédent.

## 16. Logs d'entraînement

Le Monitor reçoit les champs listés dans
[MONITOR_INFO_KEYWORDS](../src/train.py#L27). Les catégories principales sont :

- succès et cause de fin ;
- progression courante/maximale et commande moyenne ;
- fractions avance/maintien/recul ;
- vraie erreur finale position/rotation ;
- force, couple, maxima, impulsion et sécurité ;
- mode courant/suivant ;
- latch et déclencheur <code>contact_search</code> ;
- compte, durée, déclencheur et échec de récupération ;
- offsets résiduel/admittance ;
- reward dense, terminal et coût d'offset, courant et cumulé.

Une ligne CSV correspond à **un épisode terminé**, pas à une décision. Les
scalaires TensorBoard <code>assembly/*</code> écrits par
[ConsoleProgressCallback](../src/train.py#L91) reflètent le dernier
<code>info</code> reçu ; les scalaires <code>rollout/*</code> de SB3 sont des
moyennes glissantes. Il ne faut pas les interpréter de la même manière.

## 17. Évaluation

[evaluate.py](../src/evaluate.py) utilise par défaut :

- 10 épisodes ;
- seeds 100, 101, … ;
- la même configuration d'environnement que l'entraînement ;
- une politique déterministe ;
- le viewer MuJoCo humain.

Si le modèle demandé est absent,
[find_latest_model](../src/evaluate.py#L134) prend le fichier ZIP le plus récent
dans <code>models/</code> ou <code>checkpoints/</code> de la pièce. Ce choix est
fondé sur la date du fichier, pas sur le reward ni sur la compatibilité de
l'observation.

Les sorties sont :

~~~text
data/output/<part>/evaluation.json
data/output/<part>/evaluation.csv
~~~

Le CSV contient une ligne par épisode. Le JSON conserve les épisodes complets,
les moyennes, les compteurs de déclenchement et les sommes des composantes de
reward. Voir [_write_episode_csv](../src/evaluate.py#L190) et le
[résumé](../src/evaluate.py#L388).

La baseline sans SAC est
[evaluate_nominal.py](../src/evaluate_nominal.py). Sa
[nominal_config](../src/evaluate_nominal.py#L80) force les six variabilités à
zéro et désactive <code>contact_search</code>/<code>recovery</code>. L'action
est toujours :

~~~python
np.zeros(7, dtype=np.float32)
~~~

En <code>tracking</code>, cela signifie « aucun résiduel et progression
nominale », pas « robot immobile ».

## 18. Compatibilité des modèles

Les artefacts actuels mélangent plusieurs contrats :

| Artefact | Observation | Gamma | Compatibilité source actuelle |
|---|---:|---:|---|
| Ancien SAC libre | 38 | historique | non |
| Modèle SAC_10 évalué | 288 | 0,99 | non |
| Métadonnées SAC_11 v5 | 440 | 0,999 | non |
| Source v6 / SAC_12 | 448 | 0,999 | oui |

Le checkpoint v6 inspecté confirme :

- Stable-Baselines3 2.9.0 ;
- PyTorch 2.13.0+cu130 ;
- observation 448 ;
- action 7 ;
- buffer 50 000 ;
- batch 256 ;
- learning starts 5 000 ;
- gamma 0,999.

L'ajout d'une valeur d'observation ou un changement de sémantique d'action rend
un ancien modèle impropre à l'évaluation, même si le chargement ZIP réussit
partiellement. Après chaque changement de contrat, il faut repartir de zéro et
évaluer un checkpoint explicitement compatible.

## 19. Tests

La suite actuelle comporte 44 tests :

- chargement, normalisation de <code>s</code> et interpolation :
  [test_place_path.py](../src/test_place_path.py) ;
- modes, progression, offsets, observation, vraie pose, reward, sécurité et
  récupérations :
  [test_recovery_logic.py](../src/test_recovery_logic.py).

Parmi les comportements explicitement testés :

- action nulle = progression nominale en tracking ;
- six résiduels disponibles à toute progression ;
- contact à n'importe quel <code>s</code> = latch tactile ;
- wrench inertiel sans contact = aucun latch ;
- progression maximale récompensée une seule fois ;
- avance impossible en recovery ;
- sortie uniquement après dégagement ;
- admittance opposée au wrench et inactive sans contact ;
- transformation du wrench vers le monde ;
- seuils durs inclusifs ;
- succès impossible si la vraie pose reste fausse ;
- observation 56×8.

## 20. Limites et écarts par rapport au plan

La liste suivante est importante pour ne pas attribuer au système des
propriétés qu'il n'a pas :

- aucune variabilité de pose active aujourd'hui ;
- aucune variation de friction ;
- aucun bruit ou biais F/T aléatoire ;
- vraie pose cachée à l'acteur **et** au critic ;
- historique aplati, pas de réseau récurrent ;
- aucune normalisation adaptative des observations ou rewards ;
- pas de pénalité directe de durée de contact ;
- impulsion mesurée mais non utilisée dans le reward ;
- pas de recul physique d'urgence après un seuil dur ;
- corridor seulement translationnel ;
- pas de limite explicite finale de vitesse du portique ;
- replay buffer non sauvegardé ;
- pas de reprise automatique ;
- pas de meilleur modèle ni d'évaluation périodique durant l'apprentissage ;
- un seul environnement, pas d'entraînement parallèle ;
- pas de comparaison automatisée des quatre baselines prévues ;
- hyperparamètres SB3 incomplets dans <code>training_metadata.json</code> ;
- perturbations physiques non enregistrées dans les métriques ;
- dépendances SB3/Gymnasium non versionnées ;
- le service TensorBoard pointe encore vers
  <code>/data/output/tensorboard</code>, alors que les nouveaux runs sont sous
  <code>/data/output/&lt;part&gt;/tensorboard</code>.

## 21. Commandes usuelles

Entraîner une pièce depuis zéro :

~~~bash
ASSEMBLY_PART=part_1 TOTAL_TIMESTEPS=100000 docker compose run --rm train
~~~

Évaluer le modèle/checkpoint le plus récent avec viewer :

~~~bash
ASSEMBLY_PART=part_1 docker compose run --rm evaluate
~~~

Évaluer sans viewer :

~~~bash
ASSEMBLY_PART=part_1 EVAL_RENDER=none docker compose run --rm evaluate
~~~

Tester le suivi nominal sans SAC :

~~~bash
ASSEMBLY_PART=part_1 NOMINAL_EPISODES=5 docker compose run --rm evaluate-nominal
~~~

Après une première validation réussie, éviter le deuxième chargement SDF créé
par <code>check_env</code> :

~~~bash
SKIP_ENV_CHECK=1 ASSEMBLY_PART=part_1 docker compose run --rm train
~~~
