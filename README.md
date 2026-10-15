# trailnutri

Planificateur de nutrition pour trail / course à pied : à partir d'un GPX et d'un temps cible,
produit un découpage en segments, des temps de passage et un plan de prises de glucides.

> **Avertissement** : outil d'aide à la planification uniquement. Ce n'est **ni un conseil
> médical, ni un conseil diététique**. Adapte toujours ta stratégie à tes propres sensations,
> à ton expérience à l'entraînement et, si besoin, à l'avis d'un professionnel de santé.

## Installation

```bash
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
```

## Lancer l'interface

```bash
.venv\Scripts\python -m streamlit run app.py
```

Sans GPX importé, l'interface utilise `data/exemple.gpx`. Les réglages de la barre latérale
surchargent `config.yaml` pour la session uniquement.

## Tests

```bash
.venv\Scripts\python -m pytest
```

## Pipeline du moteur (100 % déterministe, paramètres dans `config.yaml`)

1. `gpx_loader` : lecture GPX, distance cumulée (haversine), rééchantillonnage uniforme (~10 m),
   lissage de l'altitude (Savitzky-Golay). Refuse un GPX sans altitude ou trop court.
2. `segments` : découpage en segments de ~150 m, D+/D- et pente moyenne par segment.
3. `pacing` : facteur de pente issu du coût énergétique de Minetti (plancher en descente),
   dérive de fatigue optionnelle, puis mise à l'échelle pour coller au temps cible.
   Refuse un temps cible impliquant une allure à plat hors de [2:30 ; 20:00] min/km.
4. `nutrition` : une prise toutes les X min (après le délai de départ, jamais sur la ligne
   d'arrivée), quantité = g/h × X/60, km et D+ cumulé interpolés. Export CSV (`;`, décimale `,`).
5. `plot` : profil altimétrique plotly avec marqueurs de prises (zones dures prévues).

## Feuille de route

- [x] MVP : GPX + segments + pacing, glucides à intervalle fixe, profil, interface Streamlit
- [ ] Score de difficulté par segment et zones dures
- [ ] Décalage des prises (avant montée, hors descente technique, ravitos officiels)
- [ ] Export PDF

Trace d'exemple : `data/exemple.gpx` (régénérable via `python scripts/generer_gpx_exemple.py`).
