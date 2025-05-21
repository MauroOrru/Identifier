from dataclasses import dataclass
from typing import Dict, Set, Tuple, Hashable, Optional   
from graphviz import Digraph

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
    
    def to_png(self, filename: str = "dfa", directory: str = "result") -> None:
        dot = Digraph(format='png', directory=directory)
        dot.attr(rankdir='LR')

        # Nodo iniziale invisibile
        dot.node('', shape="none")
        dot.edge('', str(self.q0))

        for state in self.Q:
            shape = "doublecircle" if state in self.F else "circle"
            dot.node(str(state), shape=shape)

        for (state, symbol), target in self.delta.items():
            dot.edge(str(state), str(target), label=str(symbol))

        output_path = dot.render(filename=filename, cleanup=True)
        print(f"DFA salvato in {output_path}")


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

def build_dfa(
    states: Set[State],
    alphabet: Set[Symbol],
    transitions: Dict[Tuple[State, Symbol], State],
    initial_state: State,
    final_states: Set[State]
) -> DFA:
    return DFA(
        Q=states,
        Sigma=alphabet,
        delta=transitions,
        q0=initial_state,
        F=final_states
    )

