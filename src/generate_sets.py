"""
A = parole generate e accettate
G = parole generate ma NON accettate
N = parole NON generate (transizione mancante)

Verifica inoltre la condizione:
    N ∩ pref(A ∪ G) = ∅
(se fallisce, lancia AssertionError).
"""

from typing import Set
from itertools import accumulate
from dfa_utils import simulate, all_words
from dfa_examples import dfa_simple, DFA


# ────────────────────────────────────────────────────────────────────────
def prefixes(w: str) -> Set[str]:
    return set("".join(w[:i]) for i in range(1, len(w) + 1))

def derive_sets(dfa: DFA, max_len: int = 4) -> tuple[Set[str], Set[str], Set[str]]:
    sigma = dfa.Sigma # Alphabet of the automaton
    A, G, N = set(), set(), set() # Accepted, generated, not generated
    for w in all_words(sigma, max_len):
        generated, accepted = simulate(dfa, w)
        if not generated:
            N.add(w)
        elif accepted:
            A.add(w)
        else:
            G.add(w)

    pref_AG = set().union(*(prefixes(w) for w in A.union(G)))
    assert N.isdisjoint(pref_AG), (
        "Condizione violata: esiste n ∈ N prefisso di parola in A ∪ G"
    )
    return A, G, N

if __name__ == "__main__":
    dfa = dfa_simple()
    A, G, N = derive_sets(dfa, max_len=4)
    print("A =", A)
    print("G =", G)
    print("N =", N)
