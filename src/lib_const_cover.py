from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from tree_builder import PrefixTree, PrefixTreeNode
import numpy as np
from dfa_examples import DFA
from typing import Dict, List, Optional, Set, Tuple, Union
import subprocess

def char(all_nodes, pi):
    char_v=[]
    for x in all_nodes:
        if x in pi:
            char_v.append(1)
        else:
            char_v.append(0)
    
    return np.array(char_v)

def export_dfa_dot(dfa: DFA, file: Union[str, Path]) -> None:
    lines: List[str] = ["digraph ReducedDFA {", "  rankdir=LR;"]
    for q in dfa.Q:
        shape = "doublecircle" if q in dfa.F else "circle"
        lines.append(f'  "{q}" [shape={shape}];')
    for (q, s), q2 in dfa.delta.items():
        lines.append(f'  "{q}" -> "{q2}" [label="{s}"];')
    lines.append("}")
    Path(file).write_text("\n".join(lines))
    print(f"[reduction] .dot saved in {file}")


def dot_to_png(dot_path: str | Path, png_path: str | Path) -> None:
    dot_path = str(dot_path)
    png_path = str(png_path)

    try:
        subprocess.run(["dot", "-Tpng", dot_path, "-o", png_path], check=True)
        print(f"PNG generated: {png_path}")
    except subprocess.CalledProcessError:
        print("Error in the conversion with Graphviz.")


def reduce_negative_set(N: Set[str]) -> Set[str]:
    N = sorted(N, key=len) 
    reduced = set()
    for word in N:
        if not any(word.startswith(w) and word != w for w in reduced):
            reduced.add(word)
    return reduced

##########################################################################
# Functions for creating the dynamic minimal cover
##########################################################################

def splitter(pi, sigma, pi_prime):
    pi_in=set()
    pi_out=set()
    
    for x in pi:
        if sigma in x.edges:
            step = x.edges[sigma]
            if step in pi_prime:
                pi_in.add(step)
            else:
                pi_out.add(step)
    
    if (len(pi_in)>0) and (len(pi_out)>0):
        decision = -1
    elif (len(pi_in)>0):
        decision = 1
    else:
        decision=0
    
    return decision

def split(pi, sigma, pi_prime):
    pi_1=set()
    pi_2=set()

    for x in pi:
        if sigma not in x.edges:
            pi_1.add(x)
            pi_2.add(x)
        else:
            step=x.edges[sigma]
            if step in pi_prime:
                pi_1.add(x)
            else:
                pi_2.add(x)

    pi_1=frozenset(pi_1)
    pi_2=frozenset(pi_2)
    return pi_1, pi_2

def update2(V, cover, pi_new, alphabet):
    if pi_new in cover:
        print("Possible error")
    
    cover.append(pi_new)
    V_new=set()

    for pi in cover:
        for sigma in alphabet:
            for pi_prime in cover:
                result=splitter(pi, sigma, pi_prime)
                if result==-1:
                    V_new.add( (pi, sigma, pi_prime) )
                if result == 1:
                    continue
        
    V.update(V_new)

    return V, cover

def compute_cover(tree, cover):
    U=[]
    all_nodes = []
    alphabet=tree.alphabet
    for node in tree.iter_nodes():
        all_nodes.append(node) # Generate a list of all nodes in the tree
        U.append(0) # Initialize U with zeros. It's length is equal to the cardinality of T
    
    U=np.array(U)

    for pi in cover:
        U=U + char(all_nodes, pi)

    V = set()
    for pi in cover:
        for sigma in alphabet:
            V_spt = set()
            for pi_prime in cover:
                result = splitter(pi, sigma, pi_prime)
                if result == -1:
                    V_spt = {(pi, sigma, pi_prime)}  
                if result == 1:
                    V_spt = set()
                    continue
            V.update(V_spt)
    
    while len(V)>0:
        
        random.shuffle(cover)
        pi, sigma, pi_prime = V.pop()

        temp=V.copy()
        for pi_bar, sigma_bar, pi_bar_prime in V:
            if pi_bar==pi:
                temp.remove((pi_bar, sigma_bar, pi_bar_prime))
        
        V=temp

        cover.remove(pi)

        U=U-char(all_nodes, pi)

        pi_1, pi_2 = split(pi, sigma, pi_prime)

        condition1=np.all( char(all_nodes, pi_1) <= (U+char(all_nodes, pi_2)))
        
        if not condition1 :
            V,cover = update2(V, cover, pi_1, alphabet)
            U=U+char(all_nodes, pi_1)

        condition2=np.all(char(all_nodes, pi_2) <= U)
        if not condition2:
            V,cover = update2(V, cover, pi_2, alphabet)
            U=U+char(all_nodes, pi_2)

    return cover

##########################################################################
# Debugging functions
##########################################################################


def check_cover_redundancy(cover: List[Set[PrefixTreeNode]]) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Check if the cover has redundant cells (one contained in the other).

    Args:
    cover: List of partitions (each partition is a set of states/nodes).

    Returns:
    (True, []) if the cover has no redundancies.
    (False, [ (i, j) ]) if cell j is contained in i.
    """
    problemi = []
    for i, pi in enumerate(cover):
        for j, pj in enumerate(cover):
            if i != j and pj.issubset(pi):
                problemi.append((i, j))
    return (len(problemi) == 0, problemi)

def check_dynamic_consistency(cover: List[Set], alphabet: Set[str]) -> Tuple[bool, List[Tuple[Set, str, str]]]:
    """
    Tests the dynamic consistency of a cover.

    Args:
    cover: List of partitions (each partition is a set of states/nodes)
    alphabet: Set of events (sigma)

    Returns:
    (True, []) if the cover is dynamically consistent.
    (False, [ (pi, sigma, reason) ]) otherwise.
    """
    problemi = []

    for pi in cover:
        for sigma in alphabet:
            destinazioni = set()

            for stato in pi:
                if sigma in stato.edges:
                    destinazioni.add(stato.edges[sigma])

            if destinazioni:
                # Find a cell that contains all the destinations
                found = False
                for pi_prime in cover:
                    if destinazioni.issubset(pi_prime):
                        found = True
                        break
                
                if not found:
                    problemi.append((pi, sigma, f"The destinations {destinazioni} are not entirely contained in any cell of the cover."))

    if problemi:
        return False, problemi
    else:
        return True, []

def print_cover(cover: List[Set[PrefixTreeNode]]) -> None:
    for idx, part in enumerate(cover):
        parole = sorted([n.word for n in part])
        print(f"Cover[{idx+1}] = {{{', '.join(parole)}}}")

import random

##########################################################################
# Functions to get the trimmed DFA
##########################################################################
def build_reduced_dfa_from_dynamic_cover(
    tree: PrefixTree,
    Cdyn: list[frozenset[PrefixTreeNode]]
) -> DFA:
    blocks = [frozenset(pi) for pi in Cdyn]

    # block map → state name
    block_names: dict[frozenset[PrefixTreeNode], str] = {}
    for i, b in enumerate(blocks):
        block_names[b] = f"q{i}"

    node_to_block: dict[PrefixTreeNode, list[frozenset[PrefixTreeNode]]] = defaultdict(list)
    for b in blocks:
        for n in b:
            node_to_block[n].append(b)

    root_blocks = node_to_block[tree.root]
    if not root_blocks:
        raise ValueError("The root of the tree does not belong to any block!")
    start_block = root_blocks[0]
    q0 = block_names[start_block]

    F: set[str] = set()
    for b in blocks:
        if any(n.label == "A" for n in b):
            F.add(block_names[b])

    Sigma = tree.alphabet

    # transitions from the prefix tree
    tr: dict[tuple[PrefixTreeNode, str], PrefixTreeNode] = {}
    for n in tree.iter_nodes():
        for sym, child in n.edges.items():
            tr[(n, sym)] = child

    delta: dict[tuple[str, str], str] = {}
    bfs = deque([start_block])
    visited = {start_block}

    while bfs:
        blk = bfs.popleft()
        src = block_names[blk]
        for sym in Sigma:
            succ = {tr[(x, sym)] for x in blk if (x, sym) in tr and tr[(x, sym)].label != "N"}
            if not succ:
                continue
            tgt_candidates = [b for b in blocks if succ.issubset(b)]
            random.shuffle(tgt_candidates)
            if not tgt_candidates:
                raise ValueError(f"Broken coherence: no block for δ({src}, '{sym}')")
            tgt = tgt_candidates[0]
            delta[(src, sym)] = block_names[tgt]
            if tgt not in visited:
                visited.add(tgt)
                bfs.append(tgt)

    Q = {block_names[b] for b in visited}

    print("Initial is ", q0)
    return DFA(Q=Q, Sigma=Sigma, delta=delta, q0=q0, F=F)

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