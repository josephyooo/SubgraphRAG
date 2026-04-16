# Rebuilding `retrieve/data_files/`

The `retrieve/data_files/` directory is **not committed** (~18 GB of
pre-computed WebQSP entity/relation embeddings and cached samples). You need
to regenerate it locally before training or evaluating the retriever.

Follow [`retrieve/README.md`](retrieve/README.md), section _"1-1: Entity and
Relation Embedding Pre-Computation"_. In short:

```bash
# 1. Create the text-encoder env (gte-large-en-v1.5)
conda create -n gte_large_en_v1-5 python=3.10 -y
conda activate gte_large_en_v1-5
pip install -r requirements/gte_large_en_v1-5.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121

# 2. Pre-compute embeddings for WebQSP
python emb.py -d webqsp
```

That populates `retrieve/data_files/` with everything the retriever expects.
Then follow section _"1-2: Retriever Development"_ to set up the retriever
env and run training/inference.
