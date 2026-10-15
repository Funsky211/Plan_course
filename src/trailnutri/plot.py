"""Graphiques plotly : profil altimétrique annoté et allure prévue."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from trailnutri.temps import formater_mmss
from trailnutri.waypoints import Ravito

# Palette (rôles, pas de couleur en dur ailleurs dans le module)
COULEUR_PROFIL = "#2a78d6"
COULEUR_PRISE = "#2a78d6"
COULEUR_RAVITO = "#4a3aa7"
COULEUR_MARCHE = "#eda100"
# Niveaux de difficulté : vert -> orange -> rouge (couleur de trait, remplissage)
COULEURS_NIVEAU = {
    "facile": ("#1f9d63", "rgba(31, 157, 99, 0.25)"),
    "modéré": ("#e08a00", "rgba(224, 138, 0, 0.28)"),
    "dur": ("#d63c3b", "rgba(214, 60, 59, 0.32)"),
}
COULEUR_TEXTE_SECONDAIRE = "#52514e"
COULEUR_GRILLE = "#e6e5e0"


def profil_altimetrique(
    trace: pd.DataFrame,
    plan: pd.DataFrame | None = None,
    segments: pd.DataFrame | None = None,
    ravitos: Sequence[Ravito] | None = None,
) -> go.Figure:
    """Figure altitude = f(km).

    trace : sortie de charger_gpx. plan : sortie de plan_glucides (marqueurs).
    segments : sortie de calculer_difficulte ; si fournie, le profil est coloré par niveau
    et le survol affiche le score. ravitos : lignes verticales.
    """
    km = trace["distance_m"] / 1000
    alt = trace["altitude_m"]

    fig = go.Figure()
    if segments is None:
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
    else:
        _profil_par_niveau(fig, trace, segments)

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
                    "<b>%{customdata[0]}</b> · %{customdata[1]:.0f} g de glucides<br>"
                    "km %{x:.2f} · D+ %{customdata[2]:.0f} m<br>"
                    "%{customdata[3]}<extra></extra>"
                ),
            )
        )

    for ravito in ravitos or []:
        fig.add_vline(
            x=ravito.km, line={"color": COULEUR_RAVITO, "width": 1, "dash": "dot"},
            annotation={"text": ravito.nom, "textangle": -90, "font": {"size": 10}},
            annotation_position="top left",
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


def _profil_par_niveau(fig: go.Figure, trace: pd.DataFrame, segments: pd.DataFrame) -> None:
    """Une portion de profil par zone, colorée selon son niveau ; survol = score."""
    distance = trace["distance_m"].to_numpy()
    milieux = ((segments["debut_m"] + segments["fin_m"]) / 2).to_numpy()
    score = np.interp(distance, milieux, segments["difficulte"].to_numpy())
    deja_en_legende: set[str] = set()

    debut_m, fin_m = segments["debut_m"].to_numpy(), segments["fin_m"].to_numpy()
    niveaux = segments["niveau"].to_numpy()
    debut_zone = 0
    for i in range(1, len(segments) + 1):
        if i < len(segments) and niveaux[i] == niveaux[debut_zone]:
            continue
        niv = str(niveaux[debut_zone])
        # Bornes incluses : deux zones voisines partagent un point, pas de trou dans le profil
        dedans = (distance >= debut_m[debut_zone]) & (distance <= fin_m[i - 1])
        trait, remplissage = COULEURS_NIVEAU[niv]
        fig.add_trace(
            go.Scatter(
                x=distance[dedans] / 1000,
                y=trace["altitude_m"].to_numpy()[dedans],
                name=niv.capitalize(),
                legendgroup=niv,
                showlegend=niv not in deja_en_legende,
                mode="lines",
                line={"color": trait, "width": 2},
                fill="tozeroy",
                fillcolor=remplissage,
                customdata=[(float(s), niv) for s in score[dedans]],
                hovertemplate=(
                    "km %{x:.2f} · %{y:.0f} m<br>"
                    "difficulté %{customdata[0]:.2f} (%{customdata[1]})<extra></extra>"
                ),
            )
        )
        deja_en_legende.add(niv)
        debut_zone = i


def graphique_allure(segments: pd.DataFrame) -> go.Figure:
    """Allure prévue (min/km, hors arrêts) le long de la course ; segments en marche marqués."""
    km = segments["debut_m"] / 1000
    allure_min = segments["allure_s_km"] / 60
    etiquettes = [formater_mmss(a) for a in segments["allure_s_km"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=km,
            y=allure_min,
            name="Allure prévue",
            mode="lines",
            line={"color": COULEUR_PROFIL, "width": 2, "shape": "hv"},
            customdata=list(zip(etiquettes, segments["pente"] * 100, segments["mode"],
                                strict=True)),
            hovertemplate=(
                "km %{x:.2f}<br><b>%{customdata[0]}/km</b> · pente %{customdata[1]:.0f} %"
                "<br>%{customdata[2]}<extra></extra>"
            ),
        )
    )
    marche = segments["mode"] == "marche"
    if marche.any():
        fig.add_trace(
            go.Scatter(
                x=km[marche],
                y=allure_min[marche],
                name="Marche (pente forte)",
                mode="markers",
                marker={"color": COULEUR_MARCHE, "size": 8,
                        "line": {"color": "white", "width": 2}},
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        template="plotly_white",
        hovermode="closest",
        margin={"l": 50, "r": 20, "t": 30, "b": 40},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        xaxis={"title": "Distance (km)", "gridcolor": COULEUR_GRILLE},
        # Axe inversé : plus haut = plus rapide, plus lisible pour un coureur
        yaxis={"title": "Allure (min/km)", "gridcolor": COULEUR_GRILLE, "autorange": "reversed"},
        font={"color": COULEUR_TEXTE_SECONDAIRE},
    )
    return fig
