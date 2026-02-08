#!/usr/bin/env python3
"""Check what's actually in the HuggingFace datasets."""

from datasets import load_dataset

print("="*50)
print("Checking HuggingFace Datasets Structure")
print("="*50)

# Check MBZUAI/SocialMaze
print("\n1. MBZUAI/SocialMaze:")
print("-" * 30)
try:
    socialmaze = load_dataset("MBZUAI/SocialMaze")
    print(f"Available splits: {list(socialmaze.keys())}")
    for split_name, split_data in socialmaze.items():
        print(f"\n{split_name} split:")
        print(f"  - Number of examples: {len(split_data)}")
        print(f"  - Features: {split_data.features.keys()}")
        if len(split_data) > 0:
            print(f"  - First example keys: {split_data[0].keys()}")
            # Show sample
            sample = split_data[0]
            for key, value in sample.items():
                if isinstance(value, str):
                    print(f"    - {key}: {value[:100]}...")
                else:
                    print(f"    - {key}: {value}")
except Exception as e:
    print(f"Error loading SocialMaze: {e}")

# Check bolinlai/Werewolf-Among-Us  
print("\n2. bolinlai/Werewolf-Among-Us:")
print("-" * 30)
try:
    werewolf = load_dataset("bolinlai/Werewolf-Among-Us")
    print(f"Available splits: {list(werewolf.keys())}")
    for split_name, split_data in werewolf.items():
        print(f"\n{split_name} split:")
        print(f"  - Number of examples: {len(split_data)}")
        print(f"  - Features: {split_data.features.keys()}")
        if len(split_data) > 0:
            print(f"  - First example keys: {split_data[0].keys()}")
            # Show sample
            sample = split_data[0]
            for key, value in sample.items():
                if isinstance(value, str):
                    print(f"    - {key}: {value[:100]}...")
                elif isinstance(value, list) and len(value) > 0:
                    print(f"    - {key}: {value[:3]}...")
                else:
                    print(f"    - {key}: {value}")
except Exception as e:
    print(f"Error loading Werewolf-Among-Us: {e}")

print("\n" + "="*50)
print("Dataset inspection complete!")
print("="*50)