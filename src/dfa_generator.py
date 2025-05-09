# =============================================================================
# dfa_generator.py  –  Simple DFA utilities for ACCPS Project 2
# =============================================================================
from __future__ import annotations
import random
import os
from itertools import product
from pathlib import Path
from typing import Dict, List, Set, Tuple, Iterable, Optional

State = str
Symbol = str
__all__ = [
    "DFA",
    "dfa_parity",
    "dfa_contains",
    "random_dfa",
    "compose",
    "simulate_samples",
]

# -----------------------------------------------------------------------------#
def ensure_dir(file_path: str | Path) -> None:
    """Create parent directory if it does not exist."""
    Path(file_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Core DFA class
# =============================================================================
class DFA:
    """Minimal deterministic finite automaton (complete)."""

    def __init__(
        self,
        states: Set[State],
        alphabet: Set[Symbol],
        transition: Dict[State, Dict[Symbol, State]],
        initial: State,
        accepting: Set[State],
        name: str = "DFA",
    ):
        self.Q = states
        self.Sigma = alphabet
        self.delta = transition
        self.q0 = initial
        self.F = accepting
        self.name = name

        # --- sanity checks -------------------------------------------------- #
        assert initial in states
        assert accepting.issubset(states)
        for q in states:
            assert set(transition[q].keys()) == alphabet, (
                f"State {q!r} missing symbols "
                f"{alphabet - set(transition[q].keys())}"
            )

    # --------------------------------------------------------------------- #
    def step(self, q: State, a: Symbol) -> State:
        return self.delta[q][a]

    def accepts(self, w: Iterable[Symbol] | str) -> bool:
        q = self.q0
        for a in w:
            q = self.step(q, a)
        return q in self.F

    # --------------------------------------------------------------------- #
    def to_graphviz(self, filename: str | Path) -> None:
        """Export PNG using graphviz (`pip install graphviz`, plus system pkg)."""
        try:
            import graphviz  # type: ignore
        except ImportError as e:
            raise ImportError("Missing dependency: pip install graphviz") from e

        ensure_dir(filename)
        dot = graphviz.Digraph(comment=self.name)
        dot.attr(rankdir="LR")

        dot.node("", shape="none")  # invisible start arrow
        for q in self.Q:
            shape = "doublecircle" if q in self.F else "circle"
            dot.node(q, shape=shape)

        dot.edge("", self.q0)
        for q in self.Q:
            for a, qp in self.delta[q].items():
                dot.edge(q, qp, label=a)

        dot.render(str(filename), format="png", cleanup=True)

    # --------------------------------------------------------------------- #
    def __str__(self) -> str:
        return f"<{self.name}: |Q|={len(self.Q)}, |Σ|={len(self.Sigma)}>"


# =============================================================================
# 1. Structured DFAs
# =============================================================================
def dfa_parity(symbol: Symbol = "a", modulo: int = 2) -> DFA:
    """Accetta stringhe in cui `symbol` compare un numero ≡ 0 (mod m) di volte."""
    states = {f"s{i}" for i in range(modulo)}
    alphabet = {symbol}
    delta = {
        q: {symbol: f"s{(int(q[1:]) + 1) % modulo}"} for q in states
    }
    return DFA(states, alphabet, delta, "s0", {"s0"}, name="Parity")


def dfa_contains(
    sub: str, alphabet: Optional[Set[Symbol]] = None
) -> DFA:
    """Accetta le stringhe che contengono la substring `sub` (Knuth-Morris-Pratt DFA)."""
    if alphabet is None:
        alphabet = set(sub)
    k = len(sub)
    states = {f"q{i}" for i in range(k + 1)}
    delta = {q: {} for q in states}

    # build KMP-like transition table
    for i in range(k + 1):
        for a in alphabet:
            if i < k and a == sub[i]:
                delta[f"q{i}"][a] = f"q{i+1}"
            else:
                j = i
                while j > 0 and sub[:j] != (sub[:i] + a)[-j:]:
                    j -= 1
                delta[f"q{i}"][a] = f"q{j}"

    return DFA(states, alphabet, delta, "q0", {f"q{k}"}, name=f"Contains<{sub}>")


# =============================================================================
# 2. Random DFA
# =============================================================================
def random_dfa(
    n_states: int,
    alphabet: Set[Symbol],
    acceptance_ratio: float = 0.5,
    seed: Optional[int] = None,
) -> DFA:
    """Genera un DFA completo con transizioni casuali."""
    rng = random.Random(seed)
    states = {f"r{i}" for i in range(n_states)}
    delta = {
        q: {a: rng.choice(tuple(states)) for a in alphabet} for q in states
    }
    initial = rng.choice(tuple(states))
    accepting = {q for q in states if rng.random() < acceptance_ratio}
    return DFA(states, alphabet, delta, initial, accepting, name="RandomDFA")


# =============================================================================
# 3. Concurrent composition (synchronous product)
# =============================================================================
def compose(
    dfas: List[DFA], accepting_policy: str = "intersection"
) -> DFA:
    """
    Composizione concorrente (prodotto sincrono) di più DFA.
    Sincronizza i simboli comuni, self-loop sui simboli privati.
    """
    if not dfas:
        raise ValueError("No DFA to compose.")

    Sigma = set().union(*(d.Sigma for d in dfas))
    product_states = list(product(*(d.Q for d in dfas)))
    trans: Dict[Tuple, Dict[Symbol, Tuple]] = {q: {} for q in product_states}

    for q in product_states:
        for a in Sigma:
            next_state = []
            for qi, d in zip(q, dfas):
                next_state.append(d.delta[qi][a] if a in d.Sigma else qi)
            trans[q][a] = tuple(next_state)

    if accepting_policy == "intersection":
        accepting = {
            q
            for q in product_states
            if all(qi in d.F for qi, d in zip(q, dfas))
        }
    elif accepting_policy == "union":
        accepting = {
            q
            for q in product_states
            if any(qi in d.F for qi, d in zip(q, dfas))
        }
    else:
        raise ValueError("accepting_policy must be 'intersection' or 'union'")

    initial = tuple(d.q0 for d in dfas)
    # stringify tuples for nicer display
    str_states = {str(s) for s in product_states}
    str_delta = {
        str(s): {a: str(t) for a, t in trans_s.items()}
        for s, trans_s in trans.items()
    }
    str_initial = str(initial)
    str_accepting = {str(s) for s in accepting}

    return DFA(
        str_states,
        Sigma,
        str_delta,
        str_initial,
        str_accepting,
        name="ProductDFA",
    )


# =============================================================================
# 4. Generate S⁺ / S⁻ samples
# =============================================================================
def simulate_samples(
    dfa: DFA,
    max_len: int = 6,
    n_positive: int = 100,
    n_negative: int = 100,
    seed: Optional[int] = None,
    max_attempts: int = 10_000,
) -> Tuple[List[str], List[str]]:
    """
    Estrae insiemi S⁺ e S⁻ tramite simulazione casuale.
    Limita i tentativi a `max_attempts` per evitare loop infiniti.
    """
    rng = random.Random(seed)
    Σ = list(dfa.Sigma)

    def random_word() -> str:
        return "".join(rng.choice(Σ) for _ in range(rng.randint(0, max_len)))

    S_pos, S_neg, attempts = set(), set(), 0
    while (
        (len(S_pos) < n_positive or len(S_neg) < n_negative)
        and attempts < max_attempts
    ):
        w = random_word()
        if dfa.accepts(w):
            if len(S_pos) < n_positive:
                S_pos.add(w)
        else:
            if len(S_neg) < n_negative:
                S_neg.add(w)
        attempts += 1

    if len(S_pos) < n_positive or len(S_neg) < n_negative:
        print(
            f"[WARN] solo {len(S_pos)} positivi e {len(S_neg)} negativi "
            f"trovati in {attempts} tentativi (max_len={max_len})."
        )

    return sorted(S_pos), sorted(S_neg)

    def to_dot(self) -> str:
        """Restituisce il codice Graphviz .dot come stringa."""
        lines = ['digraph {', 'rankdir=LR;']
        lines.append(f'"start" [shape=none];')
        for q in self.Q:
            shape = "doublecircle" if q in self.F else "circle"
            lines.append(f'"{q}" [shape={shape}];')
        lines.append(f'"start" -> "{self.q0}";')
        for q in self.Q:
            for a, qp in self.delta[q].items():
                lines.append(f'"{q}" -> "{qp}" [label="{a}"];')
        lines.append('}')
        return "\n".join(lines)
