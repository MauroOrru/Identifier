"""
Definisce alcuni DFA d'esempio da cui ricavare gli insiemi A, G, N.

Ogni DFA è descritto come una dataclass DFA:
    Q   : set di stati (hashable)
    Sigma: set di simboli
    delta: dict (stato, simbolo) -> stato
    q0  : stato iniziale
    F   : set di stati finali

Fornisce:
    - dfa_simple()        → DFA piccolo (Σ = {a,b})
    - dfa_with_loops()    → DFA con loop e ramo morto
    - __all__ = [nomi]    per import rilassato
"""

from dataclasses import dataclass
from typing import Dict, Set, Tuple, Hashable

State = Hashable
Symbol = Hashable

@dataclass(frozen=True)
class DFA:
    Q: Set[State]
    Sigma: Set[Symbol]
    delta: Dict[Tuple[State, Symbol], State]
    q0: State
    F: Set[State]

    # --- utilità rapide --------------------------------------------------
    def next(self, q: State, sym: Symbol) -> State | None:
        """Transizione parziale (None se non definita)."""
        return self.delta.get((q, sym))

# ────────────────────────────────────────────────────────────────────────
def dfa_simple() -> DFA:
    """DFA d’esempio (accetta le stringhe con numero pari di 'a')."""

    Q = {"s0", "s1"}                 # s0 = pari, s1 = dispari
    Sigma = {"a", "b"}
    delta = {
        ("s0", "a"): "s1",
        ("s0", "b"): "s0",
        ("s1", "a"): "s0",
        ("s1", "b"): "s1",
    }
    return DFA(Q, Sigma, delta, q0="s0", F={"s0"})

# Puoi aggiungere altri DFA qui...
__all__ = ["DFA", "dfa_simple"]
