# Project 2 ACCPS Course 2025 - Identification of DFAs by Consistent Reduction

This project implements the algorithm for identifying Deterministic Finite Automata (DFAs) through consistent reduction, based on the construction of a dynamically consistent cover.

## Project Structure

The project is organized as follows:

```bash
.
├── consistent_reduction.ipynb
├── outputs/
└── src/
    ├── dfa_examples.py
    ├── dfa_utils.py
    ├── generate_sets.py
    ├── lib_const_cover.py
    └── tree_builder.py
```

* `consistent_reduction.ipynb`: The main Jupyter notebook that demonstrates the entire DFA identification pipeline.
* `outputs/`: Directory where generated DFA graphs and prefix tree images will be saved in `.dot` and `.png` formats.
* `src/`: Contains Python modules with the implementations of various functionalities:
    * `dfa_examples.py`: Definitions of the `DFA` class and a simple DFA example.
    * `dfa_utils.py`: Utility functions for DFA simulation, word generation, random DFA creation, and concurrent composition.
    * `generate_sets.py`: Functions to derive sets $A$ (accepted), $G$ (generated but not accepted), and $N$ (not generated) from a DFA.
    * `lib_const_cover.py`: Contains implementations for reducing the set $N$, building and checking the consistent cover, and constructing the reduced DFA.
    * `tree_builder.py`: Functions for building the prefix tree from sets $A$, $G$, and $N$.

## Pipeline Description (as seen in `consistent_reduction.ipynb`)

The `consistent_reduction.ipynb` notebook illustrates the step-by-step process of DFA identification:

1.  **Reference DFA Generation/Selection**:
    * Initially, a reference DFA is generated either using `dfa_simple()` or by concurrently composing two random DFAs to create a more complex one.
    * The selected DFA is exported to `.dot` format and visualized as a `.png` image.

2.  **Derivation of String Sets**:
    * Using the `derive_sets` function, sets $A$ (accepted strings), $G$ (generated but not accepted strings), and $N$ (not generated strings) are produced from the reference DFA, up to a specified maximum length.
    * The set $N$ is subsequently reduced using `reduce_negative_set()` to remove redundancies.

3.  **Prefix Tree Construction**:
    * A `PrefixTree` is built from sets $A$, $G$, and $N$ and the DFA's alphabet. Each node in the tree is labeled according to its membership in these sets.
    * The prefix tree is exported and visualized.

4.  **Reduction Process (Consistent Cover Computation)**:
    * An initial cover (`cover0`) is constructed based on the node labels of the tree ($A$, $G$, $N$, $Y$, $Z$).
    * The `compute_cover` function refines the initial cover to obtain a dynamically consistent cover (`cover_final`). This cover is crucial for DFA reduction.
    * Dynamic consistency and non-redundancy checks are performed on the final cover.

5.  **Reduced DFA Construction**:
    * Finally, using the dynamically consistent cover (`cover_final`), the reduced DFA (`dfa_red`) representing the identified system is built.
    * The reduced DFA is exported and visualized.

## Requirements

To run the notebook and associated scripts, you need to install the dependencies listed in the `requirements.txt` file.

## Usage

1.  Clone the repository:
    ```bash
    git clone https://github.com/MauroOrru/Identifier.git
    cd <your_repository_name>
    ```
2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Ensure [Graphviz](https://graphviz.org/download/) is installed on your system, as it is used for generating `.png` images from `.dot` files.
5.  Open and run the `consistent_reduction.ipynb` notebook in Jupyter or VS Code.

## Contributions

Feel free to fork the repository, make changes, and submit pull requests.
