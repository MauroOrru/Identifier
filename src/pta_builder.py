# =============================================================================
# pta_builder.py  –  Prefix‑Tree Acceptor utilities for ACCPS Project 2
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Iterable

try:
    import graphviz  # optional, only for visualization
except ImportError:
    graphviz = None

Symbol = str
StateID = int

__all__ = [
    "PTAState",
    "PTA",
]

# -----------------------------------------------------------------------------
@dataclass
class PTAState:
    """Nodo dell'albero dei prefissi."""

    sid: StateID
    label: str = "?"  # "A", "N" oppure "?"
    children: Dict[Symbol, StateID] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"PTAState({self.sid}, '{self.label}', {list(self.children.keys())})"


# -----------------------------------------------------------------------------
class PTA:
    """Prefix‑Tree Acceptor costruito da S⁺ e S⁻ (stringhe).

    • La radice ha id = 0.
    • Ogni nodo conosce i suoi figli attraverso `children[sym] = sid`.
    • label: "A" se terminale di stringa positiva, "N" se negativa, "?" altrimenti.
    """

    def __init__(self, alphabet: Set[Symbol]):
        self.alphabet: Set[Symbol] = set(alphabet)
        self.states: Dict[StateID, PTAState] = {0: PTAState(0)}
        self.next_id: int = 1

    # .....................................................................
    def _new_state(self) -> StateID:
        sid = self.next_id
        self.states[sid] = PTAState(sid)
        self.next_id += 1
        return sid

    # .....................................................................
    def add_word(self, word: str, positive: bool) -> None:
        cur = 0
        for ch in word:
            if ch not in self.alphabet:
                raise ValueError(f"Symbol {ch!r} non nell'alfabeto {self.alphabet}")
            if ch not in self.states[cur].children:
                self.states[cur].children[ch] = self._new_state()
            cur = self.states[cur].children[ch]
        st = self.states[cur]
        if positive:
            if st.label == "N":
                raise ValueError("Conflitto: nodo marcato sia A che N")
            st.label = "A"
        else:
            if st.label == "A":
                raise ValueError("Conflitto: nodo marcato sia A che N")
            st.label = "N"

    # .....................................................................
    @classmethod
    def from_samples(cls, positives: List[str], negatives: List[str]) -> "PTA":
        alphabet = set().union(*(set(w) for w in positives + negatives))
        pta = cls(alphabet)
        for w in positives:
            pta.add_word(w, positive=True)
        for w in negatives:
            pta.add_word(w, positive=False)
        return pta

    # Alias per notebook convenienza
    build = from_samples

    # .....................................................................
    def accepts(self, word: Iterable[Symbol] | str) -> bool:
        """True se `word` arriva a un nodo marcato 'A'. Se arriva a nodo 'N' ⇒ False.
        Se finisce su '?' ⇒ consideriamo rifiutata (coerente con PTA standard)."""
        cur = 0
        for ch in word:
            if ch not in self.states[cur].children:
                return False
            cur = self.states[cur].children[ch]
        return self.states[cur].label == "A"

    # .....................................................................
    def to_graphviz(self, filename: str | Path, *, show_ids: bool = False) -> None:
        if graphviz is None:
            raise RuntimeError("Graphviz non installato; pip install graphviz")
        g = graphviz.Digraph(comment="PTA")
        g.attr(rankdir="LR")
        g.node("", shape="none")
        g.edge("", "0")
        for sid, st in self.states.items():
            shape = "doublecircle" if st.label == "A" else "circle"
            label = str(sid) if show_ids else ""
            if st.label in {"A", "N"}:
                label += f"\n{st.label}"
            g.node(str(sid), label=label, shape=shape, style="filled" if st.label != "?" else "")
            for sym, tgt in st.children.items():
                g.edge(str(sid), str(tgt), label=sym)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        g.render(str(filename), format="png", cleanup=True)

    # .....................................................................
    def to_dot(self) -> str:
        lines = ["digraph {", "rankdir=LR;"]
        lines.append('"start" [shape=none];')
        lines.append('"start" -> "0";')
        for sid, st in self.states.items():
            shape = "doublecircle" if st.label == "A" else "circle"
            label = str(sid)
            if st.label in {"A", "N"}:
                label += f"\\n{st.label}"
            lines.append(f'"{sid}" [label="{label}", shape={shape}];')
            for sym, tgt in st.children.items():
                lines.append(f'"{sid}" -> "{tgt}" [label="{sym}"];')
        lines.append("}")
        return "\n".join(lines)

    # .....................................................................
    def __str__(self) -> str:
        return f"<PTA: |Q|={len(self.states)}, |Σ|={len(self.alphabet)}>"