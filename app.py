"""Interface Streamlit minimale. Lancer : streamlit run app.py"""

from __future__ import annotations

import io
from dataclasses import replace
from datetime import time
from pathlib import Path

import streamlit as st

from trailnutri.config import charger_config
from trailnutri.gpx_loader import GpxError, charger_gpx, denivele
from trailnutri.nutrition import NutritionError, exporter_csv, plan_glucides
from trailnutri.pacing import PacingError, calculer_temps
from trailnutri.plot import profil_altimetrique
from trailnutri.segments import decouper_segments

GPX_EXEMPLE = Path(__file__).parent / "data" / "exemple.gpx"

st.set_page_config(page_title="trailnutri", layout="wide")
st.title("Plan de ravitaillement trail")
st.caption(
    "Outil d'aide à la planification : ce n'est ni un conseil médical, ni un conseil diététique."
)

cfg = charger_config()

# --- Entrées -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Course")
    fichier = st.file_uploader("Trace GPX", type=["gpx"])
    st.caption("Sans fichier, la trace d'exemple (~15 km, ~760 m D+) est utilisée.")
    col_h, col_m = st.columns(2)
    heures = col_h.number_input("Temps cible (h)", min_value=0, max_value=72, value=2)
    minutes = col_m.number_input("(min)", min_value=0, max_value=59, value=0)
    avec_depart = st.checkbox("Afficher l'heure réelle")
    heure_depart = st.time_input("Heure de départ", value=time(8, 0)) if avec_depart else None

    st.header("Athlète")
    derive = st.slider(
        "Dérive de fatigue (%)", 0, 30, int(cfg.pacing.derive_fatigue * 100),
        help="Ralentissement en fin de course par rapport au départ.",
    )
    glucides = st.slider(
        "Glucides (g/h)", 20, int(cfg.nutrition.glucides_g_h_max), int(cfg.nutrition.glucides_g_h)
    )
    intervalle = st.slider("Intervalle entre prises (min)", 10, 45, int(cfg.nutrition.intervalle_min))
    delai = st.slider(
        "Pas de prise avant (min)", 0, 30, int(cfg.nutrition.delai_premiere_prise_min)
    )

# Les réglages de la barre latérale surchargent config.yaml pour cette session uniquement
cfg_pacing = replace(cfg.pacing, derive_fatigue=derive / 100)
cfg_nutrition = replace(
    cfg.nutrition, glucides_g_h=glucides, intervalle_min=intervalle, delai_premiere_prise_min=delai
)

# --- Calcul ------------------------------------------------------------------------------
try:
    source = io.StringIO(fichier.getvalue().decode("utf-8")) if fichier else GPX_EXEMPLE
    trace = charger_gpx(source, cfg.gpx)
    segments = calculer_temps(
        decouper_segments(trace, cfg.segments), heures * 3600 + minutes * 60, cfg_pacing
    )
    plan = plan_glucides(segments, cfg_nutrition, heure_depart)
except (GpxError, PacingError, NutritionError) as erreur:
    st.error(str(erreur))
    st.stop()

# --- Résultats ---------------------------------------------------------------------------
d_plus, d_moins = denivele(trace["altitude_m"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Distance", f"{trace['distance_m'].iloc[-1] / 1000:.1f} km")
c2.metric("D+ / D-", f"{d_plus:.0f} / {d_moins:.0f} m")
c3.metric("Prises", f"{len(plan)}")
c4.metric("Glucides au total", f"{plan['glucides_g'].sum():.0f} g")

st.plotly_chart(profil_altimetrique(trace, plan), use_container_width=True)

st.subheader("Plan de ravitaillement")
st.dataframe(
    plan[["heure", "km", "d_plus_cumul_m", "glucides_g", "raison"]],
    hide_index=True,
    column_config={
        "heure": "Heure",
        "km": st.column_config.NumberColumn("Km", format="%.2f"),
        "d_plus_cumul_m": st.column_config.NumberColumn("D+ cumulé (m)", format="%.0f"),
        "glucides_g": st.column_config.NumberColumn("Glucides (g)", format="%.0f"),
        "raison": "Raison",
    },
)
st.download_button(
    "Télécharger le plan (CSV)", exporter_csv(plan), file_name="plan_ravitaillement.csv",
    mime="text/csv",
)

with st.expander("Détail des segments"):
    st.dataframe(
        segments[["debut_m", "fin_m", "pente", "d_plus_m", "allure_s_km", "temps_fin_s"]].round(3),
        hide_index=True,
    )
