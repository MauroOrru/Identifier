"""
tree_builder.py
Costruzione e gestione del prefix-tree acceptor.
Compatibile con Python ≥ 3.8
"""

from __future__ import annotations   # ok anche con 3.9
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Literal
from graphviz import Digraph

# ────────────────────────────────────────────────────────────────────────
Label = Literal["A", "G", "N", "Y", "Z"]


@dataclass
class PrefixTreeNode:
    """Nodo dell’albero dei prefissi."""
    word: str
    edges: Dict[str, "PrefixTreeNode"] = field(default_factory=dict)
    label: Optional[Label] = None

    # aggiunta/override ────────────────────────────────────────────────
    def add_child(self, sym: str, node: "PrefixTreeNode") -> None:
        self.edges[sym] = node

    def __repr__(self) -> str:
        return f"PrefixTreeNode({self.word!r}, label={self.label})"

    # ======  RENDIAMO HASHABLE  ======================================
    def __hash__(self) -> int:            # l’hash dipende solo dalla parola
        return hash(self.word)

    def __eq__(self, other) -> bool:      # due nodi uguali se stessa word
        return isinstance(other, PrefixTreeNode) and self.word == other.word
    # ==================================================================

# ────────────────────────────────────────────────────────────────────────
class PrefixTree:
    def __init__(self, alphabet: Set[str]) -> None:
        self.alphabet = alphabet
        self.root = PrefixTreeNode("")     # ε

    # inserimento parola ------------------------------------------------
    def insert_word(self, w: str) -> PrefixTreeNode:
        node = self.root
        for ch in w:
            if ch not in node.edges:
                node.add_child(ch, PrefixTreeNode(node.word + ch))
            node = node.edges[ch]
        return node

    # iterazione in profondità -----------------------------------------
    def iter_nodes(self) -> Iterable[PrefixTreeNode]:
        stack = [self.root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.edges.values())

    def export_dot(self, file: str | Path) -> None:
        lines: List[str] = ["digraph PrefixTree {", "  rankdir=LR;"]
        stack = [self.root]
        while stack:
            n = stack.pop()

            color = {
                "A": "darkgreen",
                "G": "orange",
                "N": "red",
                "Y": "blue",
                "Z": "gray",
            }.get(n.label, "black")

            shape = "doublecircle" if n.label == "A" else "circle"

            lbl = n.word if n.word else "ε"
            full_label = f"{lbl}\\n[{n.label}]"

            lines.append(
                f'  "{n.word}" [label="{full_label}", color="{color}", shape={shape}];'
            )

            for sym, child in n.edges.items():
                lines.append(f'  "{n.word}" -> "{child.word}" [label="{sym}"];')
                stack.append(child)

        lines.append("}")
        Path(file).write_text("\n".join(lines))
        print(f"[tree_builder] salvato .dot in {file}")


# ────────────────────────────────────────────────────────────────────────
def _all_prefixes(words: Set[str]) -> Set[str]:
    return {w[:i] for w in words for i in range(1, len(w) + 1)}


def build_prefix_tree(
    A: Set[str],
    G: Set[str],
    N: Set[str],
    *,
    alphabet: Set[str],
) -> PrefixTree:
    # ---- 1. insiemi Y e Z -------------------------------------------
    pref_AG = _all_prefixes(A | G)
    Y = pref_AG.difference(A | G)
    pref_N = _all_prefixes(N)
    Z = pref_N.difference(A | G | N | Y)

    # ---- 2. albero ---------------------------------------------------
    tree = PrefixTree(alphabet)
    for w in A | G | N | Y | Z:
        tree.insert_word(w)

    # ---- 3. etichette -----------------------------------------------
    def lab(w: str) -> Label:
        if w in A:
            return "A"
        if w in G:
            return "G"
        if w in N:
            return "N"
        if w in Y:
            return "Y"
        return "Z"

    for node in tree.iter_nodes():
        node.label = lab(node.word)

    return tree


# test rapido ------------------------------------------------------------
if __name__ == "__main__":
    # demo minimale
    A, G, N = {"abb", "abaa"}, {"ab"}, {"a", "bb"}
    tree = build_prefix_tree(A, G, N, alphabet={"a", "b"})
    tree.export_dot("prefix_tree.dot")
    print("tot nodi:", sum(1 for _ in tree.iter_nodes()))
