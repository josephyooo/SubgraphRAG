from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, SAGEConv

class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, edge_index, x):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

class DDE(nn.Module):
    """
    Distance-Aware DAG Embedding (Original Paper Module)
    Kept to preserve the structural features (distance to topic entity).
    """
    def __init__(self, num_rounds, num_reverse_rounds):
        super().__init__()
        self.layers = nn.ModuleList([PEConv() for _ in range(num_rounds)])
        self.reverse_layers = nn.ModuleList([PEConv() for _ in range(num_reverse_rounds)])
    
    def forward(self, topic_entity_one_hot, edge_index, reverse_edge_index):
        result_list = []
        h_pe = topic_entity_one_hot
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe)
            result_list.append(h_pe)
        
        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            h_pe_rev = layer(reverse_edge_index, h_pe_rev)
            result_list.append(h_pe_rev)
        
        return result_list

class Retriever(nn.Module):
    """
    Retriever model for SubgraphRAG.
    
    Combines semantic embeddings (optionally refined via GNN), DDE structural features,
    and topic positional encodings to score candidate triples.
    
    Args:
        emb_size: Dimension of the input entity/relation embeddings.
        topic_pe: Whether to use topic entity positional encoding.
        DDE_kwargs: Keyword arguments for the DDE module (num_rounds, num_reverse_rounds).
        gnn_num_layers: Number of GraphSAGE layers. Default is 2 (original behavior).
            Set to 0 to disable GNN and use a pure MLP + DDE retriever.
        gnn_hidden_dim: Hidden dimension for GNN layers. If None, defaults to emb_size
            (original behavior). When different from emb_size, a projection layer is added.
        gnn_dropout: Dropout rate applied after each GNN layer (except the last).
            Default is 0.0 (original behavior, no dropout).
    """
    def __init__(
        self,
        emb_size,
        topic_pe,
        DDE_kwargs,
        gnn_num_layers: int = 2,
        gnn_hidden_dim: Optional[int] = None,
        gnn_dropout: float = 0.0
    ):
        super().__init__()
        
        self.emb_size = emb_size
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(**DDE_kwargs)
        
        # --- GNN Encoder (Parameterized) ---
        # Configuration: gnn_num_layers in webqsp.yaml (or cwq.yaml) directly controls
        # the GraphSAGE depth here. The value flows through:
        #   webqsp.yaml -> load_yaml() -> config['retriever'] -> Retriever.__init__
        # 
        # len(self.convs) == gnn_num_layers (each SAGEConv is one GNN layer)
        #
        # Refines semantic embeddings using the graph structure via GraphSAGE.
        # With default parameters (gnn_num_layers=2, gnn_hidden_dim=emb_size, gnn_dropout=0.0),
        # behavior is identical to the original 2-layer GraphSAGE implementation.
        self.gnn_num_layers = gnn_num_layers
        self.gnn_hidden_dim = gnn_hidden_dim if gnn_hidden_dim is not None else emb_size
        self.gnn_dropout = nn.Dropout(gnn_dropout)
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        if self.gnn_num_layers > 0:
            # Add projection layer if hidden dim differs from emb_size
            if self.gnn_hidden_dim != emb_size:
                self.node_proj = nn.Linear(emb_size, self.gnn_hidden_dim)
                self.node_proj_back = nn.Linear(self.gnn_hidden_dim, emb_size)
            
            for _ in range(self.gnn_num_layers):
                self.convs.append(SAGEConv(self.gnn_hidden_dim, self.gnn_hidden_dim))
        # -----------------------------------

        # Calculate input size for the scoring MLP
        pred_in_size = 4 * emb_size
        if topic_pe:
            pred_in_size += 2 * 2 # Head/Tail topic indicator
        # Add DDE structural features
        pred_in_size += 2 * 2 * (DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )

    def forward(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot
    ):
        device = entity_embs.device
        
        # 1. Construct Initial Node Embeddings (Semantic)
        h_e = torch.cat([
            entity_embs,
            self.non_text_entity_emb(torch.LongTensor([0]).to(device)).expand(num_non_text_entities, -1)
        ], dim=0)

        # 2. Construct Graph Connectivity (from candidate triples)
        edge_index = torch.stack([h_id_tensor, t_id_tensor], dim=0)
        reverse_edge_index = torch.stack([t_id_tensor, h_id_tensor], dim=0)
        
        # --- GNN Forward Pass (Parameterized) ---
        # Refines the semantic embeddings (h_e) using the neighborhood.
        # With default parameters, this is equivalent to the original 2-layer GraphSAGE.
        if self.gnn_num_layers > 0:
            # Project to GNN hidden dimension if necessary
            if hasattr(self, "node_proj"):
                x = self.node_proj(h_e)
            else:
                x = h_e
            
            # Apply GNN layers
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                # ReLU after every layer EXCEPT the last layer (matches original behavior)
                if i < self.gnn_num_layers - 1:
                    x = F.relu(x)
                    x = self.gnn_dropout(x)
            
            # Project back if necessary
            if hasattr(self, "node_proj_back"):
                x = self.node_proj_back(x)
            
            # Residual connection + normalize (same as original)
            h_e = F.normalize(h_e + x, p=2, dim=1)
        else:
            # No GNN: just normalize the original h_e, no GraphSAGE refinement
            h_e = F.normalize(h_e, p=2, dim=1)
        # -----------------------------------------

        # 3. Construct Feature List for MLP
        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        # 4. Add DDE Structural Features (Distance to Topic)
        dde_list = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index)
        h_e_list.extend(dde_list)
        
        # Concatenate all features for every node
        h_e_combined = torch.cat(h_e_list, dim=1)

        # 5. Score Triples
        h_q = q_emb
        h_r = relation_embs[r_id_tensor]

        # Construct final input vector for each triple: [Query, Head_Feats, Relation, Tail_Feats]
        h_triple = torch.cat([
            h_q.expand(len(h_r), -1),
            h_e_combined[h_id_tensor],
            h_r,
            h_e_combined[t_id_tensor]
        ], dim=1)
        
        return self.pred(h_triple)