"""
Správa konfigurace experimentů ze souboru YAML a továrna inicializátorů.

Konfigurace je načtena do typovaných dataclass instancí — překlep v názvu
atributu odhalí editor okamžitě, ne až za běhu.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from src import Initializer


@dataclass
class CommonConfig:
    """Společná nastavení sdílená oběma algoritmy."""

    random_state: int
    max_iter: int


@dataclass
class KMeansConfig:
    """Nastavení specifická pro algoritmus k-means."""

    k: int
    initializer: str


@dataclass
class FuzzyCMeansConfig:
    """Nastavení specifická pro fuzzy c-means."""

    k: int
    q: float
    initializer: str


@dataclass
class ExperimentConfig:
    """Kompletní konfigurace experimentu načtená z YAML souboru."""

    common: CommonConfig
    kmeans: KMeansConfig
    fuzzy_cmeans: FuzzyCMeansConfig


def load_config(filepath: str = "config.yaml") -> ExperimentConfig:
    """Načte konfiguraci experimentu ze souboru YAML.

    Vrací typovanou instanci ``ExperimentConfig`` — přístup přes atributy
    (``cfg.kmeans.k``) místo slovníkových klíčů (``cfg["kmeans"]["k"]``).

    Parameters
    ----------
    filepath:
        Cesta k souboru YAML (výchozí: ``config.yaml`` v pracovním adresáři).

    Returns
    -------
    ExperimentConfig
        Typovaná konfigurace experimentu.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    return ExperimentConfig(
        common=CommonConfig(**raw["common"]),
        kmeans=KMeansConfig(**raw["kmeans"]),
        fuzzy_cmeans=FuzzyCMeansConfig(**raw["fuzzy_cmeans"]),
    )


def make_initializer(name: str, random_state: int | None = None) -> Initializer:
    """Továrna: převede jméno strategie inicializace na instanci ``Initializer``.

    Úkol:
        Vytvořte mapování názvů strategií na třídy inicializátorů a vraťte
        instanci příslušné třídy inicializovanou s daným ``random_state``.

        Podporované názvy (viz ``config.yaml``):
        - ``"random_uniform"``  → ``RandomUniformInit``
        - ``"forgy"``           → ``ForgyInit``
        - ``"kmeans++"``        → ``KMeansPlusPlusInit``

        Pro neznámý název vyvolejte ``ValueError`` s opisem dostupných možností.

        Vzor (registr tříd):
        ::

            registry = {
                "random_uniform": RandomUniformInit,
                "forgy": ForgyInit,
                "kmeans++": KMeansPlusPlusInit,
            }
            if name not in registry:
                raise ValueError(f"Neznámá inicializační strategie: {name}")
            return registry[name](random_state=random_state)

    Parameters
    ----------
    name:
        Název inicializační strategie — čteno z ``config.yaml``.
    random_state:
        Zárodek generátoru náhodných čísel pro reprodukovatelnost.

    Returns
    -------
    Initializer
        Instance vybrané inicializační strategie.
    """
    raise NotImplementedError(
        "Úkol: implementujte funkci make_initializer — namapujte řetězcový "
        "název strategie na instanci příslušné třídy Initializer."
    )


def validate_config(cfg: ExperimentConfig) -> None:
    """Ověří základní platnost konfigurace a vyvolá výjimku při chybě.

    Úkol (volitelný):
        Přidejte ověření, že konfigurace obsahuje požadované hodnoty
        ve správných rozsazích:
        - ``k >= 2`` (shlukování s méně než dvěma shluky nedává smysl)
        - ``q > 1``  (parametr fuzifikace musí být větší než 1)
        - ``initializer`` je jedním z povolených názvů

        Používejte ``assert`` nebo ``ValueError`` pro srozumitelné chybové zprávy.

    Parameters
    ----------
    cfg:
        Typovaná konfigurace načtená funkcí ``load_config``.
    """
    raise NotImplementedError(
        "Úkol (volitelný): implementujte validate_config — ověřte platnost "
        "konfigurace (k >= 2, q > 1, platný název inicializátoru)."
    )
