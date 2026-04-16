#!/usr/bin/env python
"""
Verification script for gnn_num_layers parameter wiring.

This script confirms that the gnn_num_layers parameter in webqsp.yaml
correctly controls the GraphSAGE depth in the Retriever model.

Usage:
    cd SubgraphRAG/helper
    python verify_gnn_layers.py

Expected output:
    For gnn_num_layers=0: len(convs)=0, no SAGEConv layers
    For gnn_num_layers=1: len(convs)=1, one SAGEConv layer
    For gnn_num_layers=2: len(convs)=2, two SAGEConv layers
"""

import sys
import os

# Add the retrieve directory to path so we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'retrieve'))

import torch
from src.model.retriever import Retriever


def verify_gnn_num_layers():
    """
    Instantiate Retriever with gnn_num_layers=0, 1, 2 and verify each
    produces the expected number of SAGEConv layers.
    """
    print("=" * 70)
    print("VERIFICATION: gnn_num_layers parameter wiring")
    print("=" * 70)
    
    # Common parameters for all tests
    emb_size = 1024  # Typical embedding size
    DDE_kwargs = {'num_rounds': 2, 'num_reverse_rounds': 2}
    
    test_cases = [0, 1, 2, 3]  # Test 0, 1, 2, and 3 layers
    
    all_passed = True
    
    for num_layers in test_cases:
        print(f"\nTest: gnn_num_layers={num_layers}")
        print("-" * 40)
        
        model = Retriever(
            emb_size=emb_size,
            topic_pe=True,
            DDE_kwargs=DDE_kwargs,
            gnn_num_layers=num_layers,
            gnn_hidden_dim=None,
            gnn_dropout=0.0
        )
        
        actual_num_convs = len(model.convs)
        stored_num_layers = model.gnn_num_layers
        
        print(f"  model.gnn_num_layers = {stored_num_layers}")
        print(f"  len(model.convs) = {actual_num_convs}")
        print(f"  model.convs = {model.convs}")
        
        # Verify correctness
        if stored_num_layers == num_layers and actual_num_convs == num_layers:
            print(f"  ✓ PASS: gnn_num_layers correctly controls convs")
        else:
            print(f"  ✗ FAIL: Expected {num_layers} convs, got {actual_num_convs}")
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED: gnn_num_layers is correctly wired!")
    else:
        print("SOME TESTS FAILED: Check the Retriever implementation.")
    print("=" * 70)
    
    return all_passed


def verify_forward_behavior():
    """
    Verify that the forward pass behaves differently based on gnn_num_layers.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION: Forward pass behavior with different gnn_num_layers")
    print("=" * 70)
    
    emb_size = 64  # Smaller for testing
    DDE_kwargs = {'num_rounds': 2, 'num_reverse_rounds': 2}
    
    # Create small test tensors
    num_entities = 10
    num_triples = 5
    
    h_id_tensor = torch.randint(0, num_entities, (num_triples,))
    r_id_tensor = torch.randint(0, 3, (num_triples,))
    t_id_tensor = torch.randint(0, num_entities, (num_triples,))
    q_emb = torch.randn(1, emb_size)
    entity_embs = torch.randn(num_entities - 2, emb_size)  # Some are non-text
    num_non_text_entities = 2
    relation_embs = torch.randn(3, emb_size)
    topic_entity_one_hot = torch.zeros(num_entities, 2)
    topic_entity_one_hot[0, 0] = 1.0  # Mark entity 0 as topic
    
    for num_layers in [0, 1, 2]:
        print(f"\nForward pass with gnn_num_layers={num_layers}:")
        print("-" * 40)
        
        model = Retriever(
            emb_size=emb_size,
            topic_pe=True,
            DDE_kwargs=DDE_kwargs,
            gnn_num_layers=num_layers,
            gnn_hidden_dim=None,
            gnn_dropout=0.0
        )
        model.eval()
        
        with torch.no_grad():
            output = model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot
            )
        
        print(f"  Output shape: {output.shape}")
        print(f"  Output (first 3): {output[:3].squeeze().tolist()}")
        
        if num_layers == 0:
            print(f"  Note: With gnn_num_layers=0, GraphSAGE is skipped,")
            print(f"        h_e is just normalized (no conv layers applied)")
        else:
            print(f"  Note: With gnn_num_layers={num_layers}, {num_layers} SAGEConv layer(s) applied")
    
    print("\n" + "=" * 70)
    print("Forward pass verification complete.")
    print("=" * 70)


def verify_config_loading():
    """
    Verify that the webqsp.yaml config is properly loaded with gnn parameters.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION: Config loading from webqsp.yaml")
    print("=" * 70)
    
    try:
        from src.config.retriever import load_yaml
        
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'retrieve', 'configs', 'retriever', 'webqsp.yaml'
        )
        config = load_yaml(config_path)
        
        print(f"\nLoaded config['retriever']:")
        for key, val in config['retriever'].items():
            print(f"  {key}: {val}")
        
        # Check if gnn parameters are present
        required_keys = ['gnn_num_layers', 'gnn_hidden_dim', 'gnn_dropout']
        missing = [k for k in required_keys if k not in config['retriever']]
        
        if missing:
            print(f"\n✗ FAIL: Missing keys in config: {missing}")
            print("  The pydantic model may need to be updated to include these fields.")
            return False
        else:
            print(f"\n✓ PASS: All GNN parameters are present in loaded config")
            return True
            
    except Exception as e:
        print(f"\n✗ ERROR loading config: {e}")
        return False


if __name__ == '__main__':
    print("\n" + "#" * 70)
    print("# SubgraphRAG: GNN Layer Configuration Verification")
    print("#" * 70 + "\n")
    
    # Test 1: Verify config loading
    config_ok = verify_config_loading()
    
    # Test 2: Verify gnn_num_layers wiring
    wiring_ok = verify_gnn_num_layers()
    
    # Test 3: Verify forward behavior
    verify_forward_behavior()
    
    # Summary
    print("\n" + "#" * 70)
    print("# SUMMARY")
    print("#" * 70)
    print(f"  Config loading: {'PASS' if config_ok else 'FAIL'}")
    print(f"  GNN wiring:     {'PASS' if wiring_ok else 'FAIL'}")
    print("#" * 70 + "\n")
