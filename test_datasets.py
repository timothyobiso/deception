#!/usr/bin/env python3
"""Test loading HuggingFace datasets."""

from transformers import AutoTokenizer
from deception_interpretability.data.hf_dataset_loaders import SocialMazeDataset, WerewolfAmongUsDataset, UnifiedHFDataset


def test_datasets():
    """Test loading datasets from HuggingFace."""
    print("Testing HuggingFace dataset loaders...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test UnifiedHFDataset
    print("\n1. Testing UnifiedHFDataset...")
    try:
        dataset = UnifiedHFDataset(tokenizer, max_length=256)
        dataset_dict = dataset.load_all_datasets()
        
        print(f"Successfully loaded datasets!")
        stats = dataset.get_statistics(dataset_dict)
        print(f"Statistics: {stats}")
        
        # Show sample
        if 'train' in dataset_dict and len(dataset_dict['train']) > 0:
            sample = dataset_dict['train'][0]
            print(f"\nSample from training set:")
            print(f"Text: {tokenizer.decode(sample['input_ids'][:50])}...")
            print(f"Deception label: {sample.get('deception_labels', 'N/A')}")
            print(f"Source: {sample.get('source', 'unknown')}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying individual datasets...")
        
        # Test SocialMaze
        print("\n2. Testing SocialMaze...")
        try:
            socialmaze = SocialMazeDataset(tokenizer, max_length=256)
            socialmaze_data = socialmaze.process_for_training()
            print(f"SocialMaze loaded: {list(socialmaze_data.keys())} splits")
        except Exception as e:
            print(f"SocialMaze error: {e}")
        
        # Test WerewolfAmongUs
        print("\n3. Testing Werewolf-AmongUs...")
        try:
            wa_dataset = WerewolfAmongUsDataset(tokenizer, max_length=256)
            wa_data = wa_dataset.process_for_training()
            print(f"Werewolf-AmongUs loaded: {list(wa_data.keys())} splits")
        except Exception as e:
            print(f"Werewolf-AmongUs error: {e}")


if __name__ == '__main__':
    test_datasets()