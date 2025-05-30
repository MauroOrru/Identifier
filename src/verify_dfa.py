"""
verify.py – Verifica che il DFA ridotto soddisfi le proprietà:
- A ⊆ Lm(R)
- G ⊆ L(R) \ Lm(R)
- N ∩ L(R) = ∅
"""

from typing import Set
from reduction import DFA
from dfa_utils import simulate

def verify_properties(dfa: DFA, A: Set[str], G: Set[str], N: Set[str]) -> bool:
    """
    Verifica le proprietà di correttezza del DFA rispetto a (A, G, N).
    Ritorna True se tutto è verificato.
    """
    ok = True

    for w in A:
        generated, accepted = simulate(dfa, w)
        if not (generated and accepted):
            print(f"[❌] A: '{w}' non è accettata")
            ok = False

    for w in G:
        generated, accepted = simulate(dfa, w)
        if not (generated and not accepted):
            print(f"[❌] G: '{w}' è accettata (o non generata)")
            ok = False

    for w in N:
        generated, _ = simulate(dfa, w)
        if generated:
            print(f"[❌] N: '{w}' è stata generata (violazione)")
            ok = False

    if ok:
        print("[✅] Tutte le proprietà verificate con successo.")
    return ok


# uso CLI rapido
if __name__ == "__main__":
    from dfa_examples import dfa_simple
    from generate_sets import derive_sets
    from tree_builder import build_prefix_tree
    from reduction import build_initial_cover, refine_cover, build_reduced_dfa

    dfa = dfa_simple()
    A, G, N = derive_sets(dfa, max_len=4)
    tree = build_prefix_tree(A, G, N, alphabet=dfa.Sigma)
    cover = refine_cover(tree, build_initial_cover(tree))
    dfa_red = build_reduced_dfa(tree, cover)

    verify_properties(dfa_red, A, G, N)
