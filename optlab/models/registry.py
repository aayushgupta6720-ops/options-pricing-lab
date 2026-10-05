"""Every pricing model behind one interface, so pages and tests can loop over them.

A new model (Heston, later) becomes available everywhere by adding one entry here.
"""

from collections.abc import Callable
from dataclasses import dataclass

from optlab.contracts import OptionSpec
from optlab.models import binomial, black_scholes, monte_carlo


@dataclass(frozen=True)
class Model:
    name: str
    price: Callable[[OptionSpec], float]
    american: bool  # can it price early exercise?
    description: str


MODELS: dict[str, Model] = {
    m.name: m
    for m in (
        Model("Black-Scholes", black_scholes.price_spec, False, "Closed-form price under lognormal spot."),
        Model(
            "Binomial (CRR)",
            lambda spec: binomial.price_spec(spec, steps=1000),
            True,
            "1,000-step Cox-Ross-Rubinstein tree; handles early exercise.",
        ),
        Model(
            "Monte Carlo",
            lambda spec: monte_carlo.price_spec(spec, n_paths=200_000, seed=7).price,
            False,
            "200,000 simulated terminal prices (fixed seed so results repeat).",
        ),
    )
}


def models_for(spec: OptionSpec) -> list[Model]:
    """The models that can price this option."""
    return [m for m in MODELS.values() if spec.style == "european" or m.american]
