"""The inputs every pricer takes, in one immutable object."""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

Kind = Literal["call", "put"]
Style = Literal["european", "american"]


@dataclass(frozen=True)
class OptionSpec:
    """A single vanilla option.

    Rates and the dividend yield are continuously compounded; T is in years; sigma is annualised
    (0.2 means 20%).
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    kind: Kind = "call"
    style: Style = "european"

    def __post_init__(self):
        if self.kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {self.kind!r}")
        if self.style not in ("european", "american"):
            raise ValueError(f"style must be 'european' or 'american', got {self.style!r}")
        if self.S <= 0 or self.K <= 0:
            raise ValueError("S and K must be positive")
        if self.T < 0 or self.sigma < 0:
            raise ValueError("T and sigma cannot be negative")

    def bump(self, **changes) -> "OptionSpec":
        """A copy with some fields changed, e.g. spec.bump(S=spec.S * 1.01)."""
        return replace(self, **changes)

    def payoff(self, spot):
        """Exercise value at the given spot price(s)."""
        spot = np.asarray(spot, dtype=float)
        return np.maximum(spot - self.K, 0.0) if self.kind == "call" else np.maximum(self.K - spot, 0.0)
