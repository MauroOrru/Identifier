"""
reduction.py – Consistent reduction of a PTA into a DFA (Project 2)
==================================================================
A *pedagogical* but functional implementation inspired by the algorithm
of Cai‑Giua‑Seatzu (2022).  For didactic clarity si usa un approccio di
**partition‑refinement** simile a Hopcroft, sufficiente per convalidare
il progetto su casi di test medi.  Non è l’implementazione O(n⁴) del
paper, ma produce un DFA coerente con S⁺ / S⁻ e spesso minimale.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from pta_builder import PTA, PTAState

###############################################################################
# Data‑structures per il DFA ridotto
###############################################################################
@dataclass
class DFA:
    states: Set[int]
    alphabet: Set[str]
    delta: Dict[int, Dict[str, int]]
    initial: int
    accepting: Set[int]

    def accepts(self, word: str) -> bool:
        q = self.initial
        for ch in word:
            q = self.delta[q][ch]
        return q in self.accepting

    # --- export minimal visual ------------------------------------------------
    def to_dot(self) -> str:
        lines = ["digraph {", "rankdir=LR;"]
        lines.append('"start" [shape=none];')
        lines.append(f'"start" -> "{self.initial}";')
        for q in self.states:
            shape = "doublecircle" if q in self.accepting else "circle"
            lines.append(f'"{q}" [shape={shape}];')
        for q in self.states:
            for a, q2 in self.delta[q].items():
                lines.append(f'"{q}" -> "{q2}" [label="{a}"];')
        lines.append("}")
        return "\n".join(lines)

###############################################################################
# Consistent reduction via partition refinement
###############################################################################

def reduce(pta: PTA) -> DFA:
    """Riduce una PTA a un DFA coerente con S⁺/S⁻ usando refinements iterativi.

    1. Crea la partizione iniziale P0 = {Acc, NonAcc}
    2. Iterativamente refina finché, per ogni blocco B e simbolo σ,
       gli stati di B transitano tutti nello stesso blocco.
    3. Ogni blocco finale diventa uno **stato** del DFA ridotto.
    """

    # --- 1. partizione iniziale ---------------------------------------------
    acc_block: Set[int] = {sid for sid, st in pta.states.items() if st.label == "A"}
    nonacc_block: Set[int] = set(pta.states.keys()) - acc_block
    partition: List[Set[int]] = [acc_block, nonacc_block]

    # mappa stato → indice blocco
    state2block: Dict[int, int] = {}
    for i, B in enumerate(partition):
        for s in B:
            state2block[s] = i

    changed = True
    while changed:
        changed = False
        new_partition: List[Set[int]] = []
        for B in partition:
            # sottopartiziona B in base ai vettori di destinazione per ogni simbolo
            buckets: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
            for s in B:
                sig = tuple(
                    state2block[pta.states[s].children.get(a, s)]  # self‑loop se assente
                    for a in sorted(pta.alphabet)
                )
                buckets[sig].add(s)
            # se buckets>1 abbiamo spaccato
            new_partition.extend(buckets.values())
            if len(buckets) > 1:
                changed = True
        partition = new_partition
        # aggiorna mappa stato→blocco
        state2block = {s: i for i, B in enumerate(partition) for s in B}

    # --- 3. costruisci il DFA ------------------------------------------------
    n_blocks = len(partition)
    alphabet = pta.alphabet
    delta: Dict[int, Dict[str, int]] = {
        i: {a: state2block[pta.states[next(iter(partition[i]))].children.get(a, next(iter(partition[i])))]
            for a in alphabet}
        for i in range(n_blocks)
    }
    initial = state2block[0]
    accepting = {state2block[s] for s in acc_block}
    states = set(range(n_blocks))
    return DFA(states, alphabet, delta, initial, accepting)

###############################################################################
# Convenience API
###############################################################################
__all__ = ["DFA", "reduce"]

# -------------------------------------------------------------------
# Alias espliciti per evitare conflitti/ambiguità d'importazione
# (alcuni ambienti hanno già un modulo chiamato "reduction")
# -------------------------------------------------------------------
def reduce_pta(pta: PTA) -> DFA:         # alias "parlante"
    return reduce(pta)

# compatibilità retro-attiva
__all__.extend(["reduce_pta"])
