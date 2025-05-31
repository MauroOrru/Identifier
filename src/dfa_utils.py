from typing import List, Tuple
from itertools import product
from dfa_examples import DFA
import subprocess
from pathlib import Path
from typing import Dict, Set, Tuple, Hashable, Optional
import random
from collections import deque

# -----------------------------------------------------------------------------
State = Hashable
Symbol = Hashable
Event = Hashable

def simulate(dfa: DFA, word: str) -> Tuple[bool, bool]:
    q = dfa.q0
    for ch in word:
        q_next = dfa.next(q, ch)
        if q_next is None:
            return False, False
        q = q_next
    return True, q in dfa.F

# It generate the set E star 
def all_words(sigma: set[str], max_len: int) -> list[str]:
    words = []

    current_length = 1
    while current_length <= max_len:

        def build_words(current, depth):
            if depth == 0:
                words.append("".join(current))
            else:
                for symbol in sorted(sigma):
                    build_words(current + [symbol], depth - 1)

        build_words([], current_length)
        current_length += 1

    return words

def random_dfa(
    n_states: int,
    alphabet: Set[Symbol],
    acceptance_ratio: float = 0.5,
    transition_ratio: float = 0.5,
    seed: Optional[int] = None,
) -> DFA:
    import random
    from collections import deque

    rng = random.Random(seed)
    states = [f"r{i}" for i in range(n_states)]
    state_set = set(states)
    delta: Dict[Tuple[State, Symbol], State] = {}

    # 1. Build a reachable graph from the initial state
    initial = rng.choice(states)
    reachable = {initial}
    remaining = set(states) - {initial}
    queue = deque([initial])

    # For each state, connect at least one new one (to ensure reachability)
    while remaining:
        current = queue.popleft()
        if not remaining:
            break
        target = rng.choice(list(remaining))
        symbol = rng.choice(list(alphabet))
        delta[(current, symbol)] = target
        reachable.add(target)
        remaining.remove(target)
        queue.append(target)

    # 2. Calculate the total number of transitions to insert
    max_transitions = len(states) * len(alphabet)
    target_transitions = int(max_transitions * transition_ratio)

    # 3. Add random transitions until you reach the desired number
    all_possible = [(q, a) for q in states for a in alphabet if (q, a) not in delta]
    rng.shuffle(all_possible)
    for (q, a) in all_possible:
        if len(delta) >= target_transitions:
            break
        delta[(q, a)] = rng.choice(states)

    # 4. Random accepting states
    accepting = {q for q in states if rng.random() < acceptance_ratio}

    return DFA(set(states), alphabet, delta, initial, accepting)

# -----------------------------------------------------------------------------
# Concurrent composition -------------------------------------------------------
# -----------------------------------------------------------------------------
def concurrent_composition(a: DFA, b: DFA) -> DFA:
    # Validation -----------------------------------------------------------
    if not isinstance(a, DFA) or not isinstance(b, DFA):
        raise TypeError("Both arguments to 'concurrent_composition' must be DFAs.")
    if not a.Sigma or not b.Sigma:
        raise ValueError("Automata alphabets must be non‑empty.")

    sync_events = a.Sigma & b.Sigma
    all_events = a.Sigma | b.Sigma
    initial = (a.q0, b.q0)

    states: Set[Tuple[State, State]] = {initial}
    delta: Dict[Tuple[Tuple[State, State], Event], Tuple[State, State]] = {}
    to_visit: List[Tuple[State, State]] = [initial]

    while to_visit:
        sA, sB = curr = to_visit.pop()
        for e in all_events:
            if e in sync_events:
                nA = a.step(sA, e)
                nB = b.step(sB, e)
                if nA is None or nB is None:
                    continue
                ns = (nA, nB)
            elif e in a.Sigma:
                nA = a.step(sA, e)
                if nA is None:
                    continue
                ns = (nA, sB)
            elif e in b.Sigma:
                nB = b.step(sB, e)
                if nB is None:
                    continue
                ns = (sA, nB)
            else:  # unreachable
                continue
            delta[(curr, e)] = ns
            if ns not in states:
                states.add(ns)
                to_visit.append(ns)

    finals = frozenset({(p, q) for (p, q) in states if p in a.F and q in b.F})

    return DFA(
        Q=frozenset(states),
        Sigma=frozenset(all_events),
        q0=initial,
        delta=delta,
        F=finals)