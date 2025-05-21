from dataclasses import dataclass
from typing import Dict, Set, Tuple, Hashable, Optional   

State = Hashable
Symbol = Hashable


@dataclass(frozen=True)
class DFA:
    Q: Set[State]
    Sigma: Set[Symbol]
    delta: Dict[Tuple[State, Symbol], State]
    q0: State
    F: Set[State]

    # transizione
    def next(self, q: State, sym: Symbol) -> Optional[State]:
        return self.delta.get((q, sym))


# ────────────────────────────────────────────────────────────────────────
def dfa_simple() -> DFA:
    Q = {"s0", "s1"}          
    Sigma = {"a", "b"}
    delta = {
        ("s0", "a"): "s1",
        ("s0", "b"): "s0",
        ("s1", "a"): "s0",
        ("s1", "b"): "s1",
    }
    return DFA(Q, Sigma, delta, q0="s0", F={"s0"})


__all__ = ["DFA", "dfa_simple"]
