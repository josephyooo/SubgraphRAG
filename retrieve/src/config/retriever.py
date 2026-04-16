import pydantic
import yaml

from .base import EnvYaml

class DatasetYaml(pydantic.BaseModel):
    name: str
    text_encoder_name: str

class DDEYaml(pydantic.BaseModel):
    num_rounds: int
    num_reverse_rounds: int

class RetrieverYaml(pydantic.BaseModel):
    """
    Retriever configuration for SubgraphRAG.
    
    gnn_num_layers: Number of GraphSAGE layers (in webqsp.yaml) directly controls
                    the GraphSAGE depth via the number of SAGEConv layers in 
                    Retriever.convs. Set to 0 to disable GNN entirely.
    gnn_hidden_dim: Hidden dimension for GNN layers. If None, defaults to emb_size.
    gnn_dropout: Dropout rate applied after each GNN layer (except the last).
    """
    topic_pe: bool
    DDE_kwargs: DDEYaml
    # GNN parameters - these control GraphSAGE depth and behavior
    gnn_num_layers: int = 2
    gnn_hidden_dim: int | None = None
    gnn_dropout: float = 0.0

class OptimizerYaml(pydantic.BaseModel):
    lr: float

class EvalYaml(pydantic.BaseModel):
    k_list: str

class RetrieverExpYaml(pydantic.BaseModel):
    num_epochs: int
    patience: int
    save_prefix: str

class RetrieverTrainYaml(pydantic.BaseModel):
    env: EnvYaml
    dataset: DatasetYaml
    retriever: RetrieverYaml
    optimizer: OptimizerYaml
    eval: EvalYaml
    train: RetrieverExpYaml

def load_yaml(config_file):
    with open(config_file) as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    task = yaml_data.pop('task')
    assert task == 'retriever'
    
    config = RetrieverTrainYaml(**yaml_data).model_dump()
    config['eval']['k_list'] = [
        int(k) for k in config['eval']['k_list'].split(',')]

    return config
