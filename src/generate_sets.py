from typing import Set
from itertools import accumulate
from dfa_utils import simulate, all_words
from dfa_examples import dfa_simple, DFA

def prefixes(w: str) -> Set[str]:
    return set("".join(w[:i]) for i in range(1, len(w) + 1))

def derive_sets(dfa: DFA, max_len: int = 4) -> tuple[Set[str], Set[str], Set[str]]:
    sigma = dfa.Sigma # Alphabet of the automaton
    A, G, N = set(), set(), set() # Accepted, generated, not generated
    print("Calculating total word count...")
    total = total_word_count(sigma, max_len)
    print(f"Total words to process: {total}")
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
        "Condition violated: N must not contain prefixes of A or G. "
    )
    return A, G, N

def total_word_count(sigma: set[str], max_len: int) -> int:
    k = len(sigma)
    return sum(k ** i for i in range(1, max_len + 1))
