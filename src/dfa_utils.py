"""
Funzioni di utilità per simulare un DFA e generare l'insieme di tutte le
parole fino a una certa lunghezza.

Dipendenze: solo typing.
"""

from typing import List, Tuple
from itertools import product
from dfa_examples import DFA

# ────────────────────────────────────────────────────────────────────────
def simulate(dfa: DFA, word: str) -> Tuple[bool, bool]:
    """
    Simula 'word' su 'dfa'.
    Ritorna (generated, accepted).
      generated = False se la transizione manca a un certo passo.
      accepted  = True  se la parola termina in stato finale.
    """
    q = dfa.q0
    for ch in word:
        q_next = dfa.next(q, ch)
        if q_next is None:
            return False, False
        q = q_next
    return True, q in dfa.F

# ────────────────────────────────────────────────────────────────────────
def all_words(sigma: set[str], max_len: int) -> List[str]:
    """Genera tutte le parole non vuote su 'sigma' fino a max_len."""
    words = []
    for l in range(1, max_len + 1):
        for tup in product(sorted(sigma), repeat=l):
            words.append("".join(tup))
    return words
