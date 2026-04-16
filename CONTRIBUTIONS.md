# Contributions

This fork of [Graph-COM/SubgraphRAG](https://github.com/Graph-COM/SubgraphRAG)
extends the retriever with a GraphSAGE-based Graph Neural Network encoder,
as part of the CSE 8803 Machine Learning with Graphs final project at
Georgia Tech (Fall 2025).

**Full writeup:** [`docs/CSE8803_MLG_Final_Report.pdf`](docs/CSE8803_MLG_Final_Report.pdf)

## Motivation

SubgraphRAG's baseline retriever scores each candidate triple independently
using an MLP over text and distance-to-description-entity (DDE) features.
This ignores the local connectivity structure among candidate triples: two
triples that share an entity should be able to exchange information during
retrieval. We test whether injecting GNN message passing over a
batch-induced entity graph improves retrieval quality and downstream QA
on WebQSP.

## What changed

- **`retrieve/src/model/retriever.py`** — the bulk of the change. Adds a
  configurable GraphSAGE encoder that:
  - builds a batch-induced directed entity graph from the candidate triples
    in the current minibatch,
  - applies `L ∈ {0, 1, 2}` GraphSAGE layers with mean aggregation, ReLU +
    dropout between layers, and no activation on the final layer,
  - wraps the stack in a residual connection and L2 normalization over node
    embeddings,
  - feeds the refined entity embeddings (concatenated with the relation
    embedding, question embedding, and DDE features) into the shared MLP
    scoring head.

- **`retrieve/src/config/retriever.py`** and
  **`retrieve/configs/retriever/webqsp.yaml`** — expose the new GNN depth
  hyperparameter `L` so the retriever can be swapped between MLP baseline
  (`L=0`) and GNN variants (`L=1`, `L=2`) without code changes.

- **`retrieve/train.py`** — adds early stopping on validation answer
  recall@100 (the operating point reported in the paper).

- **`reason/main.py`** — small additions to support evaluation of the
  GNN-variant retrievers end-to-end through the fixed Llama 3.1 8B Instruct
  reasoner.

- **`helper/`** (new) — analysis scripts used to produce the tables in the
  paper:
  - `analyze_results.py`, `average_results.py`, `saveresults.py` — aggregate
    per-run metrics across seeds and retrieval budgets,
  - `verify_gnn_layers.py` — sanity-checks layer counts against the saved
    configs,
  - `reasoner_results.csv`, `retrieval_results.csv`, `train_results.csv` and
    their `_summary_by_layers(_and_K).csv` aggregates — raw and summarized
    experimental outputs.

## Headline result

On WebQSP with a fixed candidate pool and reasoning pipeline:

- A **single** GraphSAGE layer (`L=1`) improves Answer Recall by **+2.0
  points at K=50** and **+1.2 points at K=100** over the MLP baseline, with
  only ~12% added retriever training cost.
- A second layer (`L=2`) gives no additional retrieval gain and introduces
  severe inference-latency instability (σ ≈ 1200s on the reasoner), which
  the paper attributes to the sparsity of batch-induced subgraphs.
- Downstream QA metrics (Hit@1, Exact Match, F1) for Llama 3.1 8B Instruct
  are unchanged, pointing to a reasoning — rather than retrieval —
  bottleneck at this recall level.

See the paper for full tables, discussion, and proposed future work.

## Data regeneration

The 18 GB `retrieve/data_files/` directory is not committed. See
[`REBUILD_DATA.md`](REBUILD_DATA.md) for how to regenerate it.
