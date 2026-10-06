# -*- coding: utf-8 -*-

"""
Created on 25. 08. 2026 at 20:59:30

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Silhouetová analýza kvality shlukování.

    Měří, jak dobře každý bod zapadá do svého shluku v porovnání s nejbližším
    sousedním shlukem. Pracuje s tvrdými popisky — kompatibilní s k-means i FCM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.distance import Distance


def silhouette_samples(
    x: np.ndarray,
    labels: np.ndarray,
    distance: "Distance",
) -> np.ndarray:
    """Vypočítá silhouetovou hodnotu pro každý bod datasetu.

    Úkol:
        Implementujte výpočet silhouetové hodnoty ``s(i)`` pro každý bod ``i``:

        .. math::
            s(i) = \\frac{b(i) - a(i)}{\\max(a(i),\\, b(i))}

        kde:

        - ``a(i)`` = průměrná vzdálenost bodu ``i`` od všech ostatních bodů
          ve **stejném** shluku (míra soudržnosti),
        - ``b(i)`` = průměrná vzdálenost bodu ``i`` od všech bodů v **nejbližším
          jiném** shluku (míra oddělenosti).

        Postup:
        1. Pro každý bod ``i`` identifikujte jeho shluk ``c = labels[i]``.
        2. Spočítejte ``a(i)`` jako průměr vzdáleností ke všem ostatním bodům
           se stejným popiskem (vylučte bod ``i`` samotný).
        3. Pro každý jiný shluk ``c'`` spočítejte průměrnou vzdálenost bodu
           ``i`` od všech bodů shluku ``c'``. Hodnota ``b(i)`` je minimum
           těchto průměrů přes všechny ``c' ≠ c``.
        4. Vraťte ``(b(i) - a(i)) / max(a(i), b(i))``.

        Speciální případ: pokud shluk obsahuje jen jeden bod, nastavte ``s(i) = 0``.

        Pro výpočet vzdáleností volejte ``distance.calculate(X[i], X[j])`` —
        zajišťuje konzistenci s metrikou použitou při shlukování.

    Parameters
    ----------
    x:
        Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
    labels:
        Tvrdé popisky shluků, tvar ``(n_bodů,)``, hodnoty 0 … k-1.
    distance:
        Instance metriky vzdálenosti — stejná jako při shlukování.

    Returns
    -------
    np.ndarray
        Silhouetové hodnoty, tvar ``(n_bodů,)``, hodnoty v [-1, 1].
        Vyšší hodnota znamená lepší zařazení bodu do shluku.
    """
    # assert: Ověřte, že x je 2D matice, labels je 1D pole stejné délky
    # a obsahuje alespoň 2 různé shluky
    assert x.ndim == 2, "Vstupní matice x musí být 2D"
    assert labels.ndim == 1, "Pole labels musí být 1D"
    assert labels.shape[0] == x.shape[0], "Délka pole labels musí odpovídat počtu bodů v x"
    assert np.unique(labels).size >= 2, "Pole labels musí obsahovat alespoň 2 různé shluky"

    silhouette = np.zeros(x.shape[0], dtype=float)
    unique_clusters = np.unique(labels)
    for i in range(x.shape[0]):
        c = labels[i]
        cluster_indices = np.where(labels == c)[0]
        if len(cluster_indices) == 1:
            a_i = 0.0
        else:
            other_in_cluster = cluster_indices[cluster_indices != i]
            distances_a = [distance.calculate(x[i], x[j]) for j in other_in_cluster]
            a_i = np.mean(distances_a)

        b_i = float('inf')
        for other_c in unique_clusters:
            if other_c == c:
                continue
            other_indices = np.where(labels == other_c)[0]
            if len(other_indices) == 0:
                continue
            distances_b = [distance.calculate(x[i], x[j]) for j in other_indices]
            mean_dist = np.mean(distances_b)
            if mean_dist < b_i:
                b_i = mean_dist

        max_ab = max(a_i, b_i)
        if max_ab == 0:
            silhouette[i] = 0.0
        else:
            silhouette[i] = (b_i - a_i) / max_ab

    return silhouette


    raise NotImplementedError(
        "Úkol: implementujte silhouette_samples() — vypočítejte silhouetovou "
        "hodnotu s(i) = (b(i) - a(i)) / max(a(i), b(i)) pro každý bod."
    )


def silhouette_score(
    x: np.ndarray,
    labels: np.ndarray,
    distance: "Distance",
) -> float:
    """Vypočítá průměrné silhouetové skóre přes všechny body.

    Úkol:
        Zavolejte ``silhouette_samples`` a vraťte průměr výsledného pole.
        Průměrné skóre slouží jako jednočíselná míra kvality shlukování —
        používá se pro výběr optimálního ``k``.

    Parameters
    ----------
    x:
        Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
    labels:
        Tvrdé popisky shluků, tvar ``(n_bodů,)``.
    distance:
        Instance metriky vzdálenosti.

    Returns
    -------
    float
        Průměrné silhouetové skóre v rozsahu [-1, 1].
        Blíže k 1 → kvalitnější shlukování.
    """

    return np.mean(silhouette_samples(x, labels, distance))
    raise NotImplementedError(
        "Úkol: implementujte silhouette_score() — vraťte průměr "
        "výstupu silhouette_samples(X, labels, distance)."
    )
