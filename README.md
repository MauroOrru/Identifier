# ACCPS Project 2 — Identification of DFAs by Consistent Reduction

This project implements a complete pipeline to reconstruct a DFA from a set of
positive (S⁺) and negative (S⁻) string samples. The core algorithm is based on
the consistent reduction method proposed in Cai–Giua–Seatzu (2022).

## Structure

```
project_root/
├── src/
│   ├── dfa_generator.py       # DFA generators and sample generators (S⁺/S⁻)
│   ├── pta_builder.py         # Prefix Tree Acceptor construction from S⁺/S⁻
│   └── reduction.py           # Consistent reduction from PTA to DFA
├── notebooks/
│   ├── 01_generate_dfa.ipynb        # Generate base DFA and samples
│   ├── 02_build_pta.ipynb           # Build PTA from S⁺/S⁻
│   └── 03_consistent_reduction.ipynb # Apply reduction and verify result
└── figs/                    # Generated DOT and PNG images
```

## Step-by-step overview

### 1. Generate training data

Using `dfa_generator.py`, we create:

* One or more DFAs (structured or random)
* The composed DFA (via synchronous product)
* Simulated sets S⁺ and S⁻ using `simulate_samples()`

### 2. Build the Prefix Tree Acceptor (PTA)

The PTA is constructed from the sample sets using `PTA.build(S_pos, S_neg)`.
Each string is inserted into a tree:

* Terminal nodes are marked "A" (accept) or "N" (reject)
* Intermediate nodes are marked "?"

### 3. Apply consistent reduction

With `reduction.reduce(pta)` we apply partition-refinement:

* Start with partition {A}, {¬A}
* Iteratively split blocks until all transitions are consistent
* Each block becomes a state in the reduced DFA

### 4. Verify consistency

The final reduced DFA is validated:

* All strings in S⁺ are accepted
* All strings in S⁻ are rejected

## Notes

* All components are written in pure Python 3.10+ with no external dependencies (except for optional `graphviz` for visualization).
* This implementation prioritizes correctness and clarity over raw performance.
* The reduction algorithm is a simplified, safe partition refinement (not O(n^4)).

## References

* Cai, K., Giua, A., & Seatzu, C. (2022). Consistent reduction of prefix trees for identification of DFAs.
* ACCPS Lab, Università di Cagliari — Tecnologie di Accesso a Sistemi Discreti
