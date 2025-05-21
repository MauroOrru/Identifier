"""
reduction.py  –  Riduzione coerente del DFA
Compatibile con Python 3.8 / 3.9
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from tree_builder import PrefixTree, PrefixTreeNode
import time
from dfa_examples import DFA


# ────────────────────────────────────────────────────────────────────────
def build_initial_cover(tree: PrefixTree) -> List[Set[PrefixTreeNode]]:
    pi1, pi2, pi3 = set(), set(), set()
    for n in tree.iter_nodes():
        if n.label == "A":
            pi1.add(n)
        elif n.label == "G":
            pi2.add(n)
        elif n.label == "N":
            pi3.add(n)
        elif n.label == "Y":
            pi1.add(n); pi2.add(n)
        elif n.label == "Z":
            pi1.add(n); pi2.add(n); pi3.add(n)
    return [pi1, pi2, pi3]

# ────────────────────────────────────────────────────────────────────────
# 2. Refinement → Dynamically Consistent Cover (versione stabile, O(n⁴))
# ────────────────────────────────────────────────────────────────────────
def refine_cover(
    tree: PrefixTree,
    cover: List[Set[PrefixTreeNode]],
    *,
    verbose: bool = False         # default NON verbose
) -> List[Set[PrefixTreeNode]]:
    """
    Raffina il cover finché diventa dinamicamente coerente
    (algoritmo O(n^4) – sufficiente per <~300 nodi).
    """
    SIGMA = tree.alphabet
    iteration = 0
    t0 = time.time()

    def signature(cov: List[Set[PrefixTreeNode]]) -> Set[frozenset]:
        return {frozenset(n.word for n in cell) for cell in cov}

    prev = signature(cover)

    while True:
        iteration += 1
        changed = False
        new_cov: List[Set[PrefixTreeNode]] = []

        for cell in cover:
            for sym in SIGMA:
                image = {n.edges[sym] for n in cell if sym in n.edges}
                if not image or any(image <= c for c in cover):
                    continue

                base = {n for n in cell if sym not in n.edges}
                inside = {n for n in cell if sym in n.edges and n.edges[sym] in image}
                outside = cell - base - inside

                if inside and outside:          # split reale
                    new_cov.extend([base | inside, base | outside])
                    changed = True
                    break
            else:
                new_cov.append(cell)
                continue
            break  # dopo uno split riparte dall'inizio

        if not changed:
            break

        now = signature(new_cov)
        if now == prev:
            if verbose:
                print(f"[refine] Nessuna variazione strutturale dopo {iteration} iterazioni.")
            break

        cover, prev = new_cov, now
        if verbose:
            print(f"[refine] Iterazione {iteration}, celle: {len(cover)}")

    if verbose:
        print(f"[refine] Finito in {iteration} iter – {round(time.time()-t0,3)}s")
    return cover





# ────────────────────────────────────────────────────────────────────────
def build_reduced_dfa(tree: PrefixTree,
                      cover: List[Set[PrefixTreeNode]]) -> DFA:
    cell_of: Dict[PrefixTreeNode, str] = {}
    Q: Set[str] = set()
    for i, c in enumerate(cover):
        name = f"q{i}"
        Q.add(name)
        for n in c:
            cell_of[n] = name

    q0 = cell_of[tree.root]
    Sigma = tree.alphabet
    delta: Dict[Tuple[str, str], str] = {}
    F: Set[str] = set()

    for c in cover:
        rep = next(iter(c))          # rappresentante
        q = cell_of[rep]
        if any(n.label == "A" for n in c):
            F.add(q)
        for sym in Sigma:
            if sym in rep.edges:
                delta[(q, sym)] = cell_of[rep.edges[sym]]

    return DFA(Q, Sigma, delta, q0, F)

# ────────────────────────────────────────────────────────────────────────
def export_dfa_dot(dfa: DFA, file: Union[str, Path]) -> None:
    lines: List[str] = ["digraph ReducedDFA {", "  rankdir=LR;"]
    for q in dfa.Q:
        shape = "doublecircle" if q in dfa.F else "circle"
        lines.append(f'  "{q}" [shape={shape}];')
    for (q, s), q2 in dfa.delta.items():
        lines.append(f'  "{q}" -> "{q2}" [label="{s}"];')
    lines.append("}")
    Path(file).write_text("\n".join(lines))
    print(f"[reduction] .dot salvato in {file}")

# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dfa_examples import dfa_simple           # demo
    from generate_sets import derive_sets         # (A,G,N)
    from tree_builder import build_prefix_tree

    dfa = dfa_simple()
    A, G, N = derive_sets(dfa, max_len=4)
    tree = build_prefix_tree(A, G, N, alphabet=dfa.Sigma)

    cover0 = build_initial_cover(tree)
    cover = refine_cover(tree, cover0)
    dfa_red = build_reduced_dfa(tree, cover)

    export_dfa_dot(dfa_red, "reduced_dfa.dot")
    print("[reduction] stati ridotti:", len(dfa_red.Q))
