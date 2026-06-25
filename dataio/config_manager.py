"""
Správa konfigurace experimentů ze souboru YAML a továrna inicializátorů.

Konfigurace je načtena do slovníku a **rozbalena v pipeline** do pojmenovaných
argumentů konstruktorů — modely nikdy nevidí slovník ani YAML.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from src.initialization import Initializer


def load_config(filepath: str = "config.yaml") -> dict:
    """Načte konfiguraci experimentu ze souboru YAML.

    Vrácený slovník má tři klíče: ``common``, ``kmeans`` a ``fuzzy_cmeans``.
    Pipeline z nich rozbalí pojmenované argumenty pro konstruktory.

    Parameters
    ----------
    filepath:
        Cesta k souboru YAML (výchozí: ``config.yaml`` v pracovním adresáři).

    Returns
    -------
    dict
        Vnořený slovník s konfigurací experimentu.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_initializer(name: str, random_state: int | None = None) -> "Initializer":
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


def validate_config(cfg: dict) -> None:
    """Ověří základní platnost konfigurace a vyvolá výjimku při chybě.

    Úkol (volitelný):
        Přidejte ověření, že konfigurace obsahuje požadované hodnoty
        ve správných rozsazích:
        - ``k >= 2`` (shlukování s méně než dvěma shluky nedává smysl)
        - ``m > 1``  (parametr fuzifikace musí být větší než 1)
        - ``initializer`` je jedním z povolených názvů

        Používejte ``assert`` nebo ``ValueError`` pro srozumitelné chybové zprávy.

    Parameters
    ----------
    cfg:
        Slovník načtený funkcí ``load_config``.
    """
    raise NotImplementedError(
        "Úkol (volitelný): implementujte validate_config — ověřte platnost "
        "konfiguračního slovníku (k >= 2, q > 1, platný název inicializátoru)."
    )
