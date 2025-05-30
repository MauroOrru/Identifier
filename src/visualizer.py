"""
visualizer.py – Visualizzazione e simulazione di un DFA interattivamente
"""

from reduction import DFA

def print_dfa(dfa: DFA) -> None:
    print("Stati:", dfa.Q)
    print("Iniziale:", dfa.q0)
    print("Finali:", dfa.F)
    print("Transizioni:")
    for (q, a), q2 in sorted(dfa.delta.items()):
        print(f"  δ({q}, {a}) → {q2}")


def simulate_path(dfa: DFA, word: str, verbose: bool = True) -> bool:
    q = dfa.q0
    if verbose:
        print(f"Simulazione per parola: '{word}'")
    for ch in word:
        next_q = dfa.next(q, ch)
        if next_q is None:
            if verbose:
                print(f"❌ transizione indefinita: δ({q}, '{ch}')")
            return False
        if verbose:
            print(f"δ({q}, '{ch}') → {next_q}")
        q = next_q
    if verbose:
        print("✅ Accettata" if q in dfa.F else "⛔ Non accettata")
    return q in dfa.F


# esempio interattivo
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

    print_dfa(dfa_red)

    while True:
        try:
            word = input("📝 Inserisci una stringa da simulare (vuoto per uscire): ")
            if not word:
                break
            simulate_path(dfa_red, word)
        except KeyboardInterrupt:
            break
