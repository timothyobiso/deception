"""Dataset loaders for social deduction game transcripts."""

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


@dataclass
class GameTranscript:
    """Represents a single game transcript."""
    game_id: str
    game_type: str
    players: List[str]
    roles: Dict[str, str]
    messages: List[Dict[str, any]]
    outcome: Optional[Dict[str, any]] = None
    metadata: Optional[Dict[str, any]] = None


class SocialDeductionDataset:
    """Base class for social deduction game datasets."""
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir or "./data/cache"
        self.raw_data = None
        self.processed_data = None
        
    def load_raw_data(self, path: str) -> None:
        """Load raw dataset from file or URL."""
        if path.endswith('.json'):
            with open(path, 'r') as f:
                self.raw_data = json.load(f)
        elif path.endswith('.csv'):
            self.raw_data = pd.read_csv(path).to_dict('records')
        else:
            self.raw_data = load_dataset(path)
    
    def preprocess(self) -> DatasetDict:
        """Preprocess raw data into training format."""
        raise NotImplementedError("Subclasses must implement preprocess method")
    
    def tokenize(self, examples: Dict) -> Dict:
        """Tokenize text examples."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided")
        
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def create_dataloaders(
        self,
        batch_size: int = 16,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        shuffle: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test dataloaders."""
        if self.processed_data is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        
        dataset_size = len(self.processed_data)
        indices = list(range(dataset_size))
        
        if shuffle:
            import random
            random.shuffle(indices)
        
        train_idx = int(train_split * dataset_size)
        val_idx = train_idx + int(val_split * dataset_size)
        
        train_indices = indices[:train_idx]
        val_indices = indices[train_idx:val_idx]
        test_indices = indices[val_idx:]
        
        train_loader = DataLoader(
            torch.utils.data.Subset(self.processed_data, train_indices),
            batch_size=batch_size,
            shuffle=shuffle
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(self.processed_data, val_indices),
            batch_size=batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            torch.utils.data.Subset(self.processed_data, test_indices),
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader


class MafiaDataset(SocialDeductionDataset):
    """Dataset for Mafia/Werewolf game transcripts."""
    
    def preprocess(self) -> DatasetDict:
        """Preprocess Mafia game transcripts."""
        processed_examples = []
        
        for game in self.raw_data:
            game_context = self._build_game_context(game)
            
            for i, message in enumerate(game['messages']):
                example = {
                    'game_id': game['game_id'],
                    'turn': i,
                    'speaker': message['speaker'],
                    'role': game['roles'].get(message['speaker'], 'unknown'),
                    'message': message['text'],
                    'context': game_context,
                    'is_deceptive': self._label_deception(message, game),
                    'game_phase': message.get('phase', 'unknown'),
                }
                processed_examples.append(example)
        
        self.processed_data = Dataset.from_list(processed_examples)
        return self.processed_data
    
    def _build_game_context(self, game: Dict) -> str:
        """Build context string for the game."""
        context = f"Game: {game['game_type']}\n"
        context += f"Players: {', '.join(game['players'])}\n"
        context += f"Phase: {game.get('current_phase', 'unknown')}\n"
        return context
    
    def _label_deception(self, message: Dict, game: Dict) -> bool:
        """Label whether a message contains deception."""
        speaker_role = game['roles'].get(message['speaker'])
        
        if speaker_role in ['mafia', 'werewolf']:
            keywords = ['innocent', 'not mafia', 'trust me', 'believe me']
            return any(keyword in message['text'].lower() for keyword in keywords)
        
        return False


class AmongUsDataset(SocialDeductionDataset):
    """Dataset for Among Us game transcripts."""
    
    def preprocess(self) -> DatasetDict:
        """Preprocess Among Us game transcripts."""
        processed_examples = []
        
        for game in self.raw_data:
            for round_data in game.get('rounds', []):
                for message in round_data.get('chat', []):
                    example = {
                        'game_id': game['game_id'],
                        'round': round_data['round_num'],
                        'speaker': message['player'],
                        'role': 'impostor' if message['player'] in game['impostors'] else 'crewmate',
                        'message': message['text'],
                        'is_emergency': round_data.get('is_emergency', False),
                        'reported_body': round_data.get('reported_body'),
                        'is_deceptive': self._label_deception(message, game),
                    }
                    processed_examples.append(example)
        
        self.processed_data = Dataset.from_list(processed_examples)
        return self.processed_data
    
    def _label_deception(self, message: Dict, game: Dict) -> bool:
        """Label whether a message contains deception."""
        if message['player'] in game.get('impostors', []):
            deception_patterns = [
                'saw them vent',
                'was with me',
                'can vouch',
                'doing tasks',
                'innocent',
            ]
            return any(pattern in message['text'].lower() for pattern in deception_patterns)
        return False


class SecretHitlerDataset(SocialDeductionDataset):
    """Dataset for Secret Hitler game transcripts."""
    
    def preprocess(self) -> DatasetDict:
        """Preprocess Secret Hitler game transcripts."""
        processed_examples = []
        
        for game in self.raw_data:
            for turn in game.get('turns', []):
                for action in turn.get('actions', []):
                    example = {
                        'game_id': game['game_id'],
                        'turn': turn['turn_num'],
                        'president': turn['president'],
                        'chancellor': turn.get('chancellor'),
                        'action_type': action['type'],
                        'player': action['player'],
                        'role': game['roles'].get(action['player']),
                        'party': game['parties'].get(action['player']),
                        'message': action.get('statement', ''),
                        'policies_drawn': turn.get('policies_drawn'),
                        'policy_enacted': turn.get('policy_enacted'),
                        'is_deceptive': self._label_deception(action, game),
                    }
                    processed_examples.append(example)
        
        self.processed_data = Dataset.from_list(processed_examples)
        return self.processed_data
    
    def _label_deception(self, action: Dict, game: Dict) -> bool:
        """Label whether an action contains deception."""
        player_party = game['parties'].get(action['player'])
        
        if player_party == 'fascist' and action['type'] == 'statement':
            if 'liberal' in action.get('statement', '').lower():
                return True
        
        return False