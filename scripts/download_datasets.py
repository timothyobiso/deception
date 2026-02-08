#!/usr/bin/env python3
"""Download and prepare social deduction game datasets."""

import os
import json
import requests
import zipfile
import tarfile
from pathlib import Path
import pandas as pd
import gdown
from typing import Dict, List
import argparse


def download_among_us_dataset(output_dir: Path):
    """Download Among Us Emergency Meeting Corpus.
    
    Paper: https://arxiv.org/abs/2309.08689
    Dataset: Contains 15,000 game discussions with impostor/crewmate labels
    """
    print("Downloading Among Us Emergency Meeting Corpus...")
    
    # The dataset is available on GitHub
    dataset_url = "https://github.com/AmongUsEmergencyMeetingCorpus/Among-Us-Emergency-Meeting-Corpus"
    
    # Create output directory
    among_us_dir = output_dir / "among_us"
    among_us_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the CSV files from the repository
    files_to_download = [
        "https://raw.githubusercontent.com/AmongUsEmergencyMeetingCorpus/Among-Us-Emergency-Meeting-Corpus/main/corpus.csv",
        "https://raw.githubusercontent.com/AmongUsEmergencyMeetingCorpus/Among-Us-Emergency-Meeting-Corpus/main/metadata.json"
    ]
    
    for url in files_to_download:
        filename = url.split("/")[-1]
        response = requests.get(url)
        if response.status_code == 200:
            with open(among_us_dir / filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {filename}")
    
    # Process into our format
    process_among_us_data(among_us_dir)
    

def download_werewolf_dataset(output_dir: Path):
    """Download Werewolf (Mafia) dataset.
    
    Dataset: Contains game logs with role assignments and chat messages
    """
    print("Downloading Werewolf/Mafia dataset...")
    
    werewolf_dir = output_dir / "werewolf"
    werewolf_dir.mkdir(parents=True, exist_ok=True)
    
    # Multiple werewolf datasets available
    # 1. One Night Ultimate Werewolf dataset
    onuw_url = "https://github.com/google-research/google-research/tree/master/one_night_ultimate_werewolf"
    
    print("Note: Werewolf dataset requires manual download from:", onuw_url)
    print("Creating sample structure for now...")
    
    # Create sample structure
    sample_data = create_werewolf_sample()
    with open(werewolf_dir / "werewolf_games.json", 'w') as f:
        json.dump(sample_data, f, indent=2)


def download_diplomacy_dataset(output_dir: Path):
    """Download Diplomacy game dataset.
    
    Contains strategic communications and deception in negotiations
    """
    print("Downloading Diplomacy dataset...")
    
    diplomacy_dir = output_dir / "diplomacy"
    diplomacy_dir.mkdir(parents=True, exist_ok=True)
    
    # The dataset from "Detecting Betrayal in Diplomacy Games"
    # Available at: https://github.com/DiplomacyTeam/Betrayal-Dataset
    
    dataset_url = "https://raw.githubusercontent.com/DiplomacyTeam/Betrayal-Dataset/main/data/games.json"
    
    try:
        response = requests.get(dataset_url, timeout=10)
        if response.status_code == 200:
            with open(diplomacy_dir / "diplomacy_games.json", 'w') as f:
                f.write(response.text)
            print("Downloaded Diplomacy dataset")
    except Exception:
        print("Could not download Diplomacy dataset, creating sample...")
        sample_data = create_diplomacy_sample()
        with open(diplomacy_dir / "diplomacy_games.json", 'w') as f:
            json.dump(sample_data, f, indent=2)


def process_among_us_data(data_dir: Path):
    """Process Among Us corpus into our format."""
    corpus_file = data_dir / "corpus.csv"
    
    if not corpus_file.exists():
        print("Corpus file not found, skipping processing")
        return
    
    df = pd.read_csv(corpus_file)
    
    # Convert to our format
    games = []
    
    # Group by game_id if available, otherwise create synthetic games
    if 'game_id' in df.columns:
        grouped = df.groupby('game_id')
    else:
        # Create synthetic game groupings
        df['synthetic_game_id'] = df.index // 20  # Group every 20 messages
        grouped = df.groupby('synthetic_game_id')
    
    for game_id, group in grouped:
        game_data = {
            'game_id': str(game_id),
            'game_type': 'among_us',
            'players': [],
            'impostors': [],
            'rounds': []
        }
        
        # Extract players and roles
        if 'player' in group.columns and 'role' in group.columns:
            players = group['player'].unique().tolist()
            impostors = group[group['role'] == 'impostor']['player'].unique().tolist()
            game_data['players'] = players
            game_data['impostors'] = impostors
        
        # Extract messages
        round_data = {
            'round_num': 1,
            'is_emergency': True,
            'chat': []
        }
        
        for _, row in group.iterrows():
            message = {
                'player': row.get('player', 'unknown'),
                'text': row.get('text', row.get('message', '')),
                'timestamp': row.get('timestamp', 0)
            }
            
            # Add role info if available
            if 'role' in row:
                message['actual_role'] = row['role']
            
            round_data['chat'].append(message)
        
        game_data['rounds'] = [round_data]
        games.append(game_data)
    
    # Save processed data
    output_file = data_dir / "among_us_processed.json"
    with open(output_file, 'w') as f:
        json.dump(games, f, indent=2)
    
    print(f"Processed {len(games)} Among Us games")
    return games


def create_werewolf_sample() -> List[Dict]:
    """Create sample Werewolf/Mafia game data."""
    games = []
    
    for i in range(5):
        game = {
            'game_id': f'werewolf_{i}',
            'game_type': 'werewolf',
            'players': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
            'roles': {
                'Alice': 'villager',
                'Bob': 'werewolf',
                'Charlie': 'villager',
                'Diana': 'seer',
                'Eve': 'werewolf',
                'Frank': 'villager'
            },
            'messages': [
                {
                    'phase': 'day',
                    'turn': 1,
                    'speaker': 'Bob',
                    'text': "I think Charlie was acting suspicious last night. We should vote him out.",
                    'timestamp': 0
                },
                {
                    'phase': 'day',
                    'turn': 1,
                    'speaker': 'Charlie',
                    'text': "What? I'm innocent! Bob is trying to deflect suspicion.",
                    'timestamp': 1
                },
                {
                    'phase': 'day',
                    'turn': 1,
                    'speaker': 'Diana',
                    'text': "I have information that suggests Eve might be a werewolf.",
                    'timestamp': 2
                },
                {
                    'phase': 'day',
                    'turn': 1,
                    'speaker': 'Eve',
                    'text': "Diana is lying! I'm just a villager. She must be the werewolf!",
                    'timestamp': 3
                },
                {
                    'phase': 'day',
                    'turn': 1,
                    'speaker': 'Alice',
                    'text': "I trust Diana. Let's vote for Eve.",
                    'timestamp': 4
                }
            ],
            'outcome': {
                'winner': 'werewolves',
                'surviving_werewolves': ['Bob', 'Eve']
            }
        }
        games.append(game)
    
    return games


def create_diplomacy_sample() -> List[Dict]:
    """Create sample Diplomacy game data."""
    games = []
    
    for i in range(3):
        game = {
            'game_id': f'diplomacy_{i}',
            'game_type': 'diplomacy',
            'players': ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey'],
            'messages': [
                {
                    'season': 'Spring 1901',
                    'from': 'Germany',
                    'to': 'France',
                    'text': "Let's work together against England. I promise not to move to Belgium.",
                    'actual_move': 'A Munich -> Belgium',  # Betrayal
                    'is_betrayal': True
                },
                {
                    'season': 'Spring 1901',
                    'from': 'France',
                    'to': 'Germany', 
                    'text': "Agreed! I'll focus on the Mediterranean.",
                    'actual_move': 'A Paris -> Picardy',
                    'is_betrayal': False
                },
                {
                    'season': 'Spring 1901',
                    'from': 'England',
                    'to': 'France',
                    'text': "Germany is planning to attack you. We should coordinate our defense.",
                    'actual_move': 'F London -> English Channel',
                    'is_betrayal': False
                }
            ],
            'outcome': {
                'winner': 'Germany',
                'final_supply_centers': {
                    'Germany': 18,
                    'France': 0,
                    'England': 5
                }
            }
        }
        games.append(game)
    
    return games


def create_dataset_info(output_dir: Path):
    """Create a dataset information file."""
    info = {
        'datasets': {
            'among_us': {
                'description': 'Among Us Emergency Meeting Corpus',
                'source': 'https://arxiv.org/abs/2309.08689',
                'features': ['impostor/crewmate labels', 'discussion text', 'voting patterns'],
                'size': '15,000 games'
            },
            'werewolf': {
                'description': 'Werewolf/Mafia game transcripts',
                'source': 'Various sources',
                'features': ['role assignments', 'day/night phases', 'elimination votes'],
                'size': 'Variable'
            },
            'diplomacy': {
                'description': 'Diplomacy negotiation and betrayal data',
                'source': 'Betrayal detection research',
                'features': ['private messages', 'public statements', 'actual moves', 'betrayal labels'],
                'size': 'Variable'
            }
        },
        'annotation_schema': {
            'deception_types': [
                'role_concealment',  # Hiding true role
                'false_accusation',  # Accusing innocent players
                'alibi_fabrication',  # Making up false alibis
                'trust_manipulation',  # Pretending to trust/distrust
                'information_withholding',  # Not sharing key information
                'misdirection',  # Redirecting suspicion
                'coalition_betrayal'  # Breaking agreements
            ],
            'labels': {
                'is_deceptive': 'Binary label for deceptive content',
                'deception_type': 'Category of deception',
                'speaker_role': 'True role of speaker',
                'claimed_role': 'Role claimed by speaker',
                'target': 'Target of deception'
            }
        }
    }
    
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("Created dataset information file")


def main():
    parser = argparse.ArgumentParser(description='Download social deduction game datasets')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory')
    parser.add_argument('--dataset', type=str, choices=['all', 'among_us', 'werewolf', 'diplomacy'],
                       default='all', help='Which dataset to download')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == 'all' or args.dataset == 'among_us':
        download_among_us_dataset(output_dir)
    
    if args.dataset == 'all' or args.dataset == 'werewolf':
        download_werewolf_dataset(output_dir)
    
    if args.dataset == 'all' or args.dataset == 'diplomacy':
        download_diplomacy_dataset(output_dir)
    
    create_dataset_info(output_dir)
    
    print(f"\nDatasets saved to {output_dir}")
    print("Run 'python scripts/train.py --data_path data/<dataset>/<file>' to start training")


if __name__ == '__main__':
    main()