# trailnutri

Planificateur de nutrition pour trail / course à pied : à partir d'un GPX et d'un temps cible,
produit un découpage en segments notés en difficulté, des temps de passage, et un plan de
ravitaillement (quand, où, combien de glucides, d'eau et de sodium, pourquoi à ce moment-là).

> **Avertissement** : outil d'aide à la planification uniquement. Ce n'est **ni un conseil
> médical, ni un conseil diététique**. Les quantités (glucides, eau, sodium) sont des valeurs
> de configuration, pas des recommandations personnalisées. Adapte toujours ta stratégie à tes
> sensations, à ton expérience à l'entraînement et, si besoin, à l'avis d'un professionnel de santé.

## Installation

```bash
py -3.10 -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -r requirements.txt   # installe aussi le package local (-e .)
```

(Sous PowerShell / cmd, remplacer `/` par `\` ; avec le venv activé, `python` suffit.)

## Lancer l'interface

```bash
python -m streamlit run app.py
```

Sans GPX importé, l'interface utilise `data/exemple.gpx` (15 km, 760 m D+, avec ravitos,
point d'eau et barrière horaire). Onglets :

| Onglet | Contenu |
|---|---|
| Plan | profil annoté (prises, zones dures, ravitos), tableau des prises, CSV, total à prévoir, barrières |
| Allure & passages | allure prévue le long du parcours (segments en marche marqués), temps de passage à chaque km |
| Difficulté | score par segment et détail des 6 facteurs |
| Points d'intérêt | POI lus dans le GPX et leur interprétation |
| Comparer | deux temps cibles côte à côte |
| Calibration | ajuste dérive de fatigue et plancher de descente à partir d'activités passées |

Le profil athlète (curseurs de la barre latérale) s'enregistre et se recharge en JSON.

## Tests et qualité

```bash
python -m pytest
python -m ruff check .
```

La CI GitHub Actions (`.github/workflows/ci.yml`) lance ruff et pytest à chaque push.

## Pipeline du moteur (100 % déterministe, paramètres dans `config.yaml`)

`config.yaml` est validé au chargement : une valeur incohérente est refusée avec un message clair.

1. `gpx_loader` : lecture GPX (traces ou routes, plusieurs segments, BOM toléré), distance
   cumulée (haversine), rééchantillonnage uniforme (~10 m), lissage de l'altitude
   (Savitzky-Golay), horodatage conservé s'il existe. Refuse un GPX illisible, sans altitude
   ou trop court.
2. `waypoints` : points d'intérêt `<wpt>` projetés sur la trace (un aller-retour donne deux
   passages), classés par mots-clés (ravito, eau, barrière, sommet, col...), heure de barrière
   lue dans le texte (« Barrière 10h30 »). Ravitos et barrières aussi saisissables à la main.
3. `segments` : découpage en segments de ~150 m, D+/D- et pente moyenne par segment.
4. `pacing` : facteur de pente issu du coût énergétique de Minetti (plancher en descente),
   dérive de fatigue, arrêts aux ravitos inclus dans le temps cible, segments raides annotés
   « marche », puis mise à l'échelle sur le temps cible. Vérifie les barrières horaires.
5. `difficulty` : score 0-1 par segment = somme pondérée de 6 facteurs (montée soutenue,
   fatigue, dernier tiers, descente technique, enchaînement de montées, creux physiologique),
   zones dures au-dessus d'un seuil.
6. `nutrition` : horaire de base (une prise toutes les X min, montée en charge en début de
   course), puis décalage de chaque prise dans une fenêtre ± X min vers le meilleur créneau
   (ravito, juste avant une montée dure, terrain roulant ; jamais en descente technique),
   quantités en grammes (chacun choisit ses produits), eau et sodium selon la température.
   Export CSV.
7. `calibration` : à partir de GPX horodatés, mesure de l'allure à plat, de la dérive de
   fatigue et du plancher de descente (pauses ignorées).
8. `pipeline` : enchaîne le tout ; `plot` : graphiques plotly.

Trace d'exemple régénérable via `python scripts/generer_gpx_exemple.py`.

## Pistes non traitées

- Export PDF (écarté pour l'instant).
- Couche LLM optionnelle pour expliquer le plan en langage naturel (hors du calcul).
