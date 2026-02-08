#!/usr/bin/env python3
"""Download social deduction game datasets from HuggingFace."""

import os
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import argparse
from typing import Dict, List, Any
from tqdm import tqdm


def download_socialmaze_dataset(output_dir: Path):
    """Download MBZUAI/SocialMaze dataset.
    
    SocialMaze: A multi-modal dataset for social deduction games including
    Among Us, Werewolf, and other social reasoning tasks.
    """
    print("Downloading MBZUAI/SocialMaze dataset...")
    
    dataset = load_dataset("MBZUAI/SocialMaze")
    
    # Create output directory
    socialmaze_dir = output_dir / "socialmaze"
    socialmaze_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split_name in dataset.keys():
        print(f"Processing {split_name} split...")
        split_data = dataset[split_name]
        
        # Convert to our format
        processed_games = []
        
        for item in tqdm(split_data):
            game_data = process_socialmaze_item(item)
            processed_games.append(game_data)
        
        # Save processed data
        output_file = socialmaze_dir / f"{split_name}_processed.json"
        with open(output_file, 'w') as f:
            json.dump(processed_games, f, indent=2)
        
        print(f"Saved {len(processed_games)} games to {output_file}")
    
    # Save dataset info
    info = {
        'dataset': 'MBZUAI/SocialMaze',
        'splits': list(dataset.keys()),
        'total_examples': sum(len(dataset[split]) for split in dataset.keys()),
        'features': list(dataset[list(dataset.keys())[0]].features.keys()) if dataset else []
    }

    with open(socialmaze_dir / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)


def process_socialmaze_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single SocialMaze item into our format."""
    processed = {
        'game_id': item.get('id', ''),
        'game_type': item.get('game_type', 'unknown'),
        'messages': [],
        'annotations': {}
    }
    
    # Extract game context
    if 'context' in item:
        processed['context'] = item['context']
    
    # Extract roles if available
    if 'roles' in item:
        processed['roles'] = item['roles']
    elif 'players' in item:
        # Try to extract roles from player info
        processed['roles'] = {}
        for player in item['players']:
            if isinstance(player, dict):
                processed['roles'][player.get('name', '')] = player.get('role', 'unknown')
    
    # Extract messages/dialogue
    if 'dialogue' in item:
        for msg in item['dialogue']:
            message = {
                'speaker': msg.get('speaker', 'unknown'),
                'text': msg.get('text', ''),
                'timestamp': msg.get('timestamp', 0),
                'phase': msg.get('phase', 'unknown')
            }
            processed['messages'].append(message)
    elif 'messages' in item:
        processed['messages'] = item['messages']
    elif 'text' in item:
        # Single text item, create a message
        processed['messages'].append({
            'speaker': item.get('speaker', 'unknown'),
            'text': item['text']
        })
    
    # Extract labels and annotations
    if 'label' in item:
        processed['annotations']['label'] = item['label']
    if 'is_deceptive' in item:
        processed['annotations']['is_deceptive'] = item['is_deceptive']
    if 'deception_type' in item:
        processed['annotations']['deception_type'] = item['deception_type']
    
    return processed


def download_werewolf_amongus_dataset(output_dir: Path):
    """Download bolinlai/Werewolf-Among-Us dataset.
    
    This dataset contains game transcripts from both Werewolf and Among Us games
    with role annotations and deception labels.
    """
    print("Downloading bolinlai/Werewolf-Among-Us dataset...")
    
    dataset = load_dataset("bolinlai/Werewolf-Among-Us")
    
    # Create output directories
    werewolf_dir = output_dir / "werewolf"
    amongus_dir = output_dir / "amongus"
    werewolf_dir.mkdir(parents=True, exist_ok=True)
    amongus_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split_name in dataset.keys():
        print(f"Processing {split_name} split...")
        split_data = dataset[split_name]
        
        werewolf_games = []
        amongus_games = []
        
        for item in tqdm(split_data):
            game_data = process_werewolf_amongus_item(item)
            
            # Separate by game type
            if 'werewolf' in game_data.get('game_type', '').lower():
                werewolf_games.append(game_data)
            elif 'among' in game_data.get('game_type', '').lower():
                amongus_games.append(game_data)
            else:
                # Try to infer from content
                if any(word in str(game_data).lower() for word in ['impostor', 'crewmate', 'vent', 'emergency']):
                    amongus_games.append(game_data)
                else:
                    werewolf_games.append(game_data)
        
        # Save Werewolf games
        if werewolf_games:
            output_file = werewolf_dir / f"{split_name}_processed.json"
            with open(output_file, 'w') as f:
                json.dump(werewolf_games, f, indent=2)
            print(f"Saved {len(werewolf_games)} Werewolf games")
        
        # Save Among Us games
        if amongus_games:
            output_file = amongus_dir / f"{split_name}_processed.json"
            with open(output_file, 'w') as f:
                json.dump(amongus_games, f, indent=2)
            print(f"Saved {len(amongus_games)} Among Us games")
    
    # Save dataset info
    info = {
        'dataset': 'bolinlai/Werewolf-Among-Us',
        'splits': list(dataset.keys()),
        'features': list(dataset[list(dataset.keys())[0]].features.keys()) if dataset else []
    }

    with open(output_dir / 'werewolf_amongus_info.json', 'w') as f:
        json.dump(info, f, indent=2)


def process_werewolf_amongus_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single Werewolf-Among-Us item."""
    processed = {
        'game_id': item.get('game_id', item.get('id', '')),
        'game_type': item.get('game_type', 'unknown'),
        'messages': [],
        'players': {},
        'annotations': {}
    }
    
    # Extract player roles
    if 'roles' in item:
        processed['players'] = item['roles']
    elif 'players' in item:
        for player in item['players']:
            if isinstance(player, dict):
                processed['players'][player.get('name', '')] = {
                    'role': player.get('role', 'unknown'),
                    'team': player.get('team', 'unknown')
                }
    
    # For Werewolf specific
    if 'werewolves' in item:
        for wolf in item['werewolves']:
            processed['players'][wolf] = {'role': 'werewolf', 'team': 'evil'}
    if 'villagers' in item:
        for villager in item['villagers']:
            if villager not in processed['players']:
                processed['players'][villager] = {'role': 'villager', 'team': 'good'}
    
    # For Among Us specific
    if 'impostors' in item:
        processed['impostors'] = item['impostors']
        for impostor in item['impostors']:
            processed['players'][impostor] = {'role': 'impostor', 'team': 'evil'}
    if 'crewmates' in item:
        processed['crewmates'] = item['crewmates']
        for crewmate in item['crewmates']:
            if crewmate not in processed['players']:
                processed['players'][crewmate] = {'role': 'crewmate', 'team': 'good'}
    
    # Extract messages
    if 'dialogue' in item:
        processed['messages'] = item['dialogue']
    elif 'messages' in item:
        processed['messages'] = item['messages']
    elif 'conversation' in item:
        # Parse conversation string
        for line in item['conversation'].split('\n'):
            if ':' in line:
                speaker, text = line.split(':', 1)
                processed['messages'].append({
                    'speaker': speaker.strip(),
                    'text': text.strip()
                })
    elif 'text' in item:
        # Single message
        processed['messages'].append({
            'speaker': item.get('speaker', 'unknown'),
            'text': item['text'],
            'role': item.get('role', 'unknown')
        })
    
    # Extract annotations
    for key in ['is_lie', 'is_deceptive', 'deception', 'lying', 'betrayal']:
        if key in item:
            processed['annotations'][key] = item[key]
    
    # Game outcome
    if 'winner' in item:
        processed['outcome'] = {'winner': item['winner']}
    elif 'outcome' in item:
        processed['outcome'] = item['outcome']
    
    return processed


def create_unified_dataset(output_dir: Path):
    """Create a unified dataset from all downloaded sources."""
    print("\nCreating unified dataset...")
    
    unified_dir = output_dir / "unified"
    unified_dir.mkdir(parents=True, exist_ok=True)
    
    all_games = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    # Collect all processed files
    for subdir in output_dir.iterdir():
        if subdir.is_dir() and subdir.name != 'unified':
            for json_file in subdir.glob('*_processed.json'):
                with open(json_file) as f:
                    games = json.load(f)
                
                # Determine split
                if 'train' in json_file.name:
                    split = 'train'
                elif 'val' in json_file.name or 'valid' in json_file.name:
                    split = 'validation'
                elif 'test' in json_file.name:
                    split = 'test'
                else:
                    # Default split based on position
                    split = 'train'
                
                # Add source info
                for game in games:
                    game['source_dataset'] = subdir.name
                
                all_games[split].extend(games)
    
    # Save unified datasets
    for split, games in all_games.items():
        if games:
            output_file = unified_dir / f"{split}.json"
            with open(output_file, 'w') as f:
                json.dump(games, f, indent=2)
            print(f"Saved {len(games)} games to {split} split")
    
    # Create statistics
    stats = {
        'total_games': sum(len(games) for games in all_games.values()),
        'splits': {split: len(games) for split, games in all_games.items()},
        'sources': {}
    }
    
    for split, games in all_games.items():
        for game in games:
            source = game.get('source_dataset', 'unknown')
            if source not in stats['sources']:
                stats['sources'][source] = 0
            stats['sources'][source] += 1
    
    with open(unified_dir / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nTotal unified dataset: {stats['total_games']} games")
    print(f"Distribution: {stats['splits']}")


def main():
    parser = argparse.ArgumentParser(description='Download social deduction datasets from HuggingFace')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for datasets')
    parser.add_argument('--dataset', type=str, 
                       choices=['all', 'socialmaze', 'werewolf-amongus'],
                       default='all',
                       help='Which dataset to download')
    parser.add_argument('--create_unified', action='store_true',
                       help='Create unified dataset from all sources')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.dataset in ['all', 'socialmaze']:
            try:
                download_socialmaze_dataset(output_dir)
            except Exception as e:
                print(f"Error downloading SocialMaze: {e}")
        
        if args.dataset in ['all', 'werewolf-amongus']:
            try:
                download_werewolf_amongus_dataset(output_dir)
            except Exception as e:
                print(f"Error downloading Werewolf-Among-Us: {e}")
        
        if args.create_unified:
            create_unified_dataset(output_dir)
        
        print(f"\nDatasets saved to {output_dir}")
        print("\nTo train a model, run:")
        print(f"python scripts/train.py --data_path {output_dir}/unified/train.json")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have the datasets library installed:")
        print("pip install datasets")


if __name__ == '__main__':
    main()