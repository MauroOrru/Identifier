"""
reduction.py  –  Riduzione coerente del DFA
Compatibile con Python 3.8 / 3.9
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
from tree_builder import PrefixTree, PrefixTreeNode
import time

@dataclass
class DFA:
    Q: Set[str]
    Sigma: Set[str]
    delta: Dict[Tuple[str, str], str]
    q0: str
    F: Set[str]

    def next(self, q: str, sym: str) -> Optional[str]:
        return self.delta.get((q, sym))

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


    pi1=frozenset(pi1)
    pi2=frozenset(pi2)
    pi3=frozenset(pi3)
    return [pi1, pi2, pi3]

def build_dfa_from_quotient(tree: PrefixTree, cover: list[set[PrefixTreeNode]]) -> DFA:
    nodes = list(tree.iter_nodes())
    Sigma = tree.alphabet
    transition = {}
    for node in nodes:
        for sym, child in node.edges.items():
            transition[(node, sym)] = child

    # 1. Blocchi come frozenset
    blocks = [frozenset(block) for block in cover]

    # 2. Mappa nodo → nome blocco
    block_names = {block: f"q{i}" for i, block in enumerate(blocks)}
    node_to_block = {}
    for block in blocks:
        for node in block:
            node_to_block[node] = block

    # 3. Stati (nomi)
    Q = set(block_names.values())

    # 4. Stato iniziale: blocco che contiene la radice
    start_block = node_to_block[tree.root]
    q0 = block_names[start_block]

    # 5. Stati finali: blocchi che contengono almeno un nodo con label "A"
    F = {
        block_names[block]
        for block in blocks
        if any(n.label == "A" for n in block)
    }

    # 6. Transizioni: (nome_blocco, simbolo) -> nome_blocco
    delta = {}
    for block in blocks:
        block_name = block_names[block]
        for node in block:
            for sym in Sigma:
                next_node = transition.get((node, sym))
                if next_node is None:
                    continue
                next_block = node_to_block.get(next_node)
                if next_block is None:
                    print(f"[WARNING] Nodo non coperto dalla cover: {next_node}")
                    continue
                delta[(block_name, sym)] = block_names[next_block]


    return DFA(Q=Q, Sigma=Sigma, delta=delta, q0=q0, F=F)

def export_dfa_dot(dfa: DFA, file: Union[str, Path]) -> None:
    lines: List[str] = ["digraph ReducedDFA {", "  rankdir=LR;"]
    for q in dfa.Q:
        shape = "doublecircle" if q in dfa.F else "circle"
        lines.append(f'  "{q}" [shape={shape}];')
    for (q, s), q2 in dfa.delta.items():
        lines.append(f'  "{q}" -> "{q2}" [label="{s}"];')
    lines.append("}")
    Path(file).write_text("\n".join(lines))


def split(pi: Set[PrefixTreeNode], sigma: str, pi_first: Set[PrefixTreeNode]) -> Tuple[Set[PrefixTreeNode], Set[PrefixTreeNode]]:
    pi1 = set()
    pi2 = set()

    for x in pi:
        if sigma not in x.edges:
            pi1 |= {x}
            pi2 |= {x}
        else:
            xi = x.edges[sigma]
            if xi in pi_first:
                pi1 |= {x}
            else:
                pi2 |= {x}

    return pi1, pi2


def splitter(
    pi: Set[PrefixTreeNode],
    sigma: str,
    pi_first: Set[PrefixTreeNode]
) -> int:
    pi_in = set()   
    pi_out = set()  

    for x in pi:
        if sigma in x.edges:
            y = x.edges[sigma]  
            # classifica x in pi_in oppure pi_out
            if y in pi_first:
                pi_in.add(y)
            else:
                pi_out.add(y)

    if len(pi_in) > 0 and len(pi_out) > 0:
        return -1
    elif len(pi_in) > 0:
        return 1
    else:
        return 0
    
def update(
    pending_splitters: Set[Tuple[Set[PrefixTreeNode], str, Set[PrefixTreeNode]]],
    cover: List[Set[PrefixTreeNode]],
    pi_new: Set[PrefixTreeNode],
    alphabet: Set[str]
) -> Tuple[
    set[Tuple[Set[PrefixTreeNode], str, Set[PrefixTreeNode]]],
    List[Set[PrefixTreeNode]]
]:
    cover.append(pi_new)

    for sigma in alphabet:
        V_split = None

        for pi_first in cover:
            val = splitter(pi_new, sigma, pi_first)
            if val == -1:
                V_split = (pi_new, sigma, pi_first)
            elif val == 1:
                V_split = None
                break

        if V_split is not None:
            V_split = (frozenset(pi_new), sigma, frozenset(pi_first))
            pending_splitters.add(V_split)

    return pending_splitters, cover

def char(all_nodes, pi):
    char_v=[]
    for x in all_nodes:
        if x in pi:
            char_v.append(1)
        else:
            char_v.append(0)
    
    return np.array(char_v)

def sum_vector(a: List[int], b: List[int]) -> List[int]:
    result = []
    result = [x + y for x, y in zip(a, b)]
    return result

def is_subset(vec_a: List[int], vec_b: List[int]) -> bool:
    for x, y in zip(vec_a, vec_b):
        if x == 1 and y == 0:
            return False
    return True

def compute_dynamically_consistent_cover(
    cover: List[Set[PrefixTreeNode]],
    alphabet: Set[str],
    tree: PrefixTree
) -> List[Set[PrefixTreeNode]]:
    
    all_nodes = []
    for node in tree.iter_nodes():
        all_nodes.append(node)

    all_nodes = [PrefixTreeNode('', label="G"), PrefixTreeNode('b', label="Y"), PrefixTreeNode('ba', label="N"), PrefixTreeNode('bb', label="G"), PrefixTreeNode('a', label="G"), PrefixTreeNode('ab', label="S")]
    
    U = [0 for _ in range(len(all_nodes))]
    
    # ROW 2
    for cell in cover:
        char_vector = char(cell, all_nodes)
        U = sum_vector(U, char_vector)

    V =  set()  # pending splitters: insieme di triple (pi, sigma, pi_target)
    i=0
    for pi in cover:
        for sigma in alphabet:
            V_spt = set()
            for pi_target in cover:
                val = splitter(pi, sigma, pi_target)
                if val == -1:
                    V_spt = {(frozenset(pi), sigma, frozenset(pi_target))}
                    break
                elif val == 1:
                    V_spt = set()
                    break
            V = V.union(V_spt)
        i += 1    

    # Row 13
    while V:
        pi_frozen, sigma, pi_target_frozen = V.pop()
        pi = set(pi_frozen)
        pi_target = set(pi_target_frozen)

        V_new = set()
        for (pi_, sigma_, pi_target_) in V:  # riga 15
            if pi_ != pi:
                V_new.add((pi_, sigma_, pi_target_))
        V = V_new

        if pi in cover:  # controllare se pi è già in cover
            cover.remove(pi)

        U_pi = char(pi, all_nodes)
        U = [u - upi for u, upi in zip(U, U_pi)]

        pi1, pi2 = split(pi, sigma, pi_target)

        char_pi1 = char(pi1, all_nodes)
        char_pi2 = char(pi2, all_nodes)

        U_plus_pi2 = sum_vector(U, char_pi2)

        if not is_subset(char_pi1, U_plus_pi2):
            # Row 23
            V, cover = update(V, cover, pi1, alphabet)
            U = sum_vector(U, char_pi1)

            for triple in list(V):
                p_check, s_check, pi_target_check = triple
                if ((pi_target_check == frozenset(pi)) and (splitter(set(p_check), s_check, pi1) == -1)):
                    V = V.union({(p_check, s_check, frozenset(pi1))})
                    V.remove(triple)

        if not is_subset(char_pi2, U): 
            V, cover = update(V, cover, pi2, alphabet)
            U = sum_vector(U, char_pi2)

            for triple in list(V):
                p_check, s_check, pi_target_check = triple
                if pi_target_check == frozenset(pi) and splitter(set(p_check), s_check, pi2) == -1:
                    V = V.union({(p_check, s_check, frozenset(pi2))})
                    V.remove(triple)

    cover = [c for c in cover if c]
    return cover

def build_reduced_dfa_from_dynamic_cover(
    tree: PrefixTree,
    Cdyn: List[Set[PrefixTreeNode]],
    N: Set[PrefixTreeNode]  # insieme dei nodi “negativi”
) -> DFA:
    # (i) Y := {0, …, len(Cdyn)-1}
    Y = set(range(len(Cdyn)))

    # (ii) y0 = blocco che contiene la radice
    x0 = tree.root
    y0 = next((y for y in Y if x0 in Cdyn[y]), None)
    if y0 is None:
        raise ValueError("Root node not found in any cell of Cdyn")

    # (iii) Ym = blocchi con almeno un nodo label 'A'
    Ym = {y for y in Y if any(x.label == "A" for x in Cdyn[y])}

    # Blocchi “da tagliare” (contengono N)
    blocked = {
        y for y in Y
        if any(n in N or n.label == "N" for n in Cdyn[y])
    }

    # (iv) tabella di transizione η
    Sigma = tree.alphabet
    delta: dict[tuple[int, str], int] = {}

    for y in Y:
        if y in blocked:                # stato irraggiungibile dall’esterno
            continue
        cell = Cdyn[y]
        for sigma in Sigma:
            for x in cell:
                child = x.edges.get(sigma)   # nessun KeyError
                if child is None:
                    continue
                # trova il blocco target che contiene child
                for y2 in Y:
                    if child in Cdyn[y2] and y2 not in blocked:
                        delta[(y, sigma)] = y2
                        break       # esci dal loop su y2
                if (y, sigma) in delta:
                    break           # esci anche dal loop su x

    return DFA(Q=Y, Sigma=Sigma, delta=delta, q0=y0, F=Ym)

