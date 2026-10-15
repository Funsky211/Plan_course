"""Profil altimétrique plotly annoté : prises de glucides et (plus tard) zones dures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Palette (rôles, pas de couleur en dur ailleurs dans le module)
COULEUR_PROFIL = "#2a78d6"
COULEUR_PRISE = "#eb6834"
COULEUR_ZONE_DURE = "#e34948"
COULEUR_TEXTE_SECONDAIRE = "#52514e"
COULEUR_GRILLE = "#e6e5e0"


def profil_altimetrique(
    trace: pd.DataFrame,
    plan: pd.DataFrame | None = None,
    zones_dures: Sequence[tuple[float, float]] | None = None,
) -> go.Figure:
    """Figure altitude = f(km).

    trace : sortie de charger_gpx. plan : sortie de plan_glucides (marqueurs).
    zones_dures : intervalles (km_debut, km_fin) surlignés (prévu pour le score de difficulté).
    """
    km = trace["distance_m"] / 1000
    alt = trace["altitude_m"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=km,
            y=alt,
            name="Altitude",
            mode="lines",
            line={"color": COULEUR_PROFIL, "width": 2},
            fill="tozeroy",
            fillcolor="rgba(42, 120, 214, 0.12)",
            hovertemplate="km %{x:.2f}<br>%{y:.0f} m<extra></extra>",
        )
    )

    if plan is not None and not plan.empty:
        # Marqueur posé sur le profil, à l'altitude du km de la prise
        alt_prises = np.interp(plan["km"], km, alt)
        fig.add_trace(
            go.Scatter(
                x=plan["km"],
                y=alt_prises,
                name="Prise de glucides",
                mode="markers",
                marker={
                    "color": COULEUR_PRISE,
                    "size": 11,
                    "line": {"color": "white", "width": 2},  # anneau pour détacher du profil
                },
                customdata=plan[["heure", "glucides_g", "d_plus_cumul_m", "raison"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> · %{customdata[1]:.0f} g<br>"
                    "km %{x:.2f} · D+ %{customdata[2]:.0f} m<br>"
                    "%{customdata[3]}<extra></extra>"
                ),
            )
        )

    for debut_km, fin_km in zones_dures or []:
        fig.add_vrect(
            x0=debut_km, x1=fin_km, fillcolor=COULEUR_ZONE_DURE, opacity=0.12, line_width=0
        )

    # Axe Y resserré sur le relief (sinon un départ à 1500 m écrase tout le profil)
    marge = max(20.0, 0.1 * (alt.max() - alt.min()))
    fig.update_layout(
        template="plotly_white",
        hovermode="closest",
        margin={"l": 50, "r": 20, "t": 30, "b": 40},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        xaxis={"title": "Distance (km)", "gridcolor": COULEUR_GRILLE},
        yaxis={
            "title": "Altitude (m)",
            "gridcolor": COULEUR_GRILLE,
            "range": [alt.min() - marge, alt.max() + marge],
        },
        font={"color": COULEUR_TEXTE_SECONDAIRE},
    )
    return fig
