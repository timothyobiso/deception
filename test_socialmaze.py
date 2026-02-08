#!/usr/bin/env python3
"""Quick test to see what's actually in SocialMaze dataset."""

from datasets import load_dataset

print("Loading MBZUAI/SocialMaze...")
dataset = load_dataset("MBZUAI/SocialMaze", streaming=True)  # Use streaming to avoid downloading everything

print(f"\nAvailable splits: {list(dataset.keys())}")

# Check what's in each split
for split_name in dataset.keys():
    print(f"\n{split_name} split:")
    
    # Look at first few examples
    examples = []
    for i, example in enumerate(dataset[split_name]):
        examples.append(example)
        if i >= 2:  # Just get 3 examples
            break
    
    if examples:
        print(f"  Features: {list(examples[0].keys())}")
        print(f"\n  First example:")
        for key, value in examples[0].items():
            if isinstance(value, str):
                print(f"    {key}: {value[:200]}...")
            else:
                print(f"    {key}: {value}")