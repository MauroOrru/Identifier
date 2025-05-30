from dataclasses import dataclass
from typing import Dict, Set, Tuple, Hashable, Optional   

State = Hashable
Symbol = Hashable
Event = Hashable


@dataclass(frozen=True)
class DFA:
    Q: Set[State]
    Sigma: Set[Symbol]
    delta: Dict[Tuple[State, Symbol], State]
    q0: State
    F: Set[State]

    # transiction function
    def next(self, q: State, sym: Symbol) -> Optional[State]:
        return self.delta.get((q, sym))
    
    def step(self, s: State, e: Event) -> State | None:
        """Return next state reached from `s` under event `e`.

        If the transition is undefined, returns *None* to avoid masking errors.
        """
        return self.delta.get((s, e))

# Starting DFA example
def dfa_simple() -> DFA:
    Q = {"s0", "s1", "s2", "s3", "s4", "s5"}          
    Sigma = {"a", "b", "c"}
    delta = {
    ("s0", "a"): "s1",
    ("s0", "c"): "s5",
    ("s1", "b"): "s2",
    ("s1", "c"): "s4",
    ("s2", "c"): "s3",
    ("s3", "a"): "s1",
}
    return DFA(Q, Sigma, delta, q0="s0", F={"s3"})

__all__ = ["DFA", "dfa_simple"]

