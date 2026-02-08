"""Loaders for real social deduction game datasets with proper annotations."""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from dataclasses import dataclass
import re


@dataclass 
class DeceptionAnnotation:
    """Annotation for deceptive behavior."""
    is_deceptive: bool
    deception_type: Optional[str] = None
    confidence: float = 1.0
    target_player: Optional[str] = None
    claimed_role: Optional[str] = None
    actual_role: Optional[str] = None


class RealAmongUsDataset:
    """Loader for the Among Us Emergency Meeting Corpus.
    
    Paper: https://arxiv.org/abs/2309.08689
    Features:
    - 15,000 games with impostor/crewmate labels
    - Discussion transcripts from emergency meetings
    - Voting patterns and elimination data
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        include_game_context: bool = True
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_game_context = include_game_context
        self.games = []
        self.processed_data = []
        
    def load_data(self):
        """Load Among Us corpus data."""
        # Try multiple file formats
        if (self.data_path / "among_us_processed.json").exists():
            with open(self.data_path / "among_us_processed.json") as f:
                self.games = json.load(f)
        elif (self.data_path / "corpus.csv").exists():
            df = pd.read_csv(self.data_path / "corpus.csv")
            self.games = self._process_csv_corpus(df)
        else:
            raise FileNotFoundError(f"No Among Us data found in {self.data_path}")
    
    def _process_csv_corpus(self, df: pd.DataFrame) -> List[Dict]:
        """Process CSV corpus into game format."""
        games = []
        
        # Detect columns
        text_col = 'text' if 'text' in df.columns else 'message'
        player_col = 'player' if 'player' in df.columns else 'speaker'
        
        # Group messages into games
        if 'game_id' in df.columns:
            grouped = df.groupby('game_id')
        else:
            # Create synthetic games
            df['game_id'] = df.index // 50  # ~50 messages per game
            grouped = df.groupby('game_id')
        
        for game_id, game_df in grouped:
            # Extract impostors and crewmates
            impostors = []
            crewmates = []
            
            if 'role' in game_df.columns:
                impostors = game_df[game_df['role'].str.lower() == 'impostor'][player_col].unique().tolist()
                crewmates = game_df[game_df['role'].str.lower() == 'crewmate'][player_col].unique().tolist()
            
            # Build game structure
            game = {
                'game_id': str(game_id),
                'impostors': impostors,
                'crewmates': crewmates,
                'messages': []
            }
            
            for _, row in game_df.iterrows():
                message = {
                    'speaker': row[player_col],
                    'text': row[text_col],
                    'role': row.get('role', 'unknown')
                }
                game['messages'].append(message)
            
            games.append(game)
        
        return games
    
    def annotate_deception(self, message: Dict, game: Dict) -> DeceptionAnnotation:
        """Annotate message with deception labels based on game context."""
        speaker = message['speaker']
        text = message['text'].lower()
        is_impostor = speaker in game.get('impostors', [])
        
        # Deception patterns for impostors
        if is_impostor:
            deception_patterns = {
                'role_concealment': [
                    'i was doing tasks', 'i was in electrical', 'saw me scan',
                    'i did wires', 'i was with', 'can vouch for me'
                ],
                'false_accusation': [
                    'saw them vent', 'they killed', 'caught them',
                    'definitely them', 'vote them out', 'sus of'
                ],
                'trust_manipulation': [
                    'trust me', 'believe me', 'i swear', 
                    'would i lie', 'you know me'
                ],
                'alibi_fabrication': [
                    'was doing', 'was at', 'was fixing', 
                    'was downloading', 'saw me at'
                ]
            }
            
            for deception_type, patterns in deception_patterns.items():
                if any(pattern in text for pattern in patterns):
                    return DeceptionAnnotation(
                        is_deceptive=True,
                        deception_type=deception_type,
                        actual_role='impostor',
                        claimed_role='crewmate'
                    )
        
        # Crewmates can also be deceptive (false accusations, wrong suspicions)
        elif any(phrase in text for phrase in ['i think', 'maybe', 'could be', 'suspicious']):
            # Check if accusation is wrong
            accused = self._extract_accused_player(text, game)
            if accused and accused not in game.get('impostors', []):
                return DeceptionAnnotation(
                    is_deceptive=False,  # Wrong but not intentionally deceptive
                    deception_type='mistaken_accusation',
                    actual_role='crewmate'
                )
        
        return DeceptionAnnotation(is_deceptive=False)
    
    def _extract_accused_player(self, text: str, game: Dict) -> Optional[str]:
        """Extract who is being accused in the message."""
        all_players = game.get('impostors', []) + game.get('crewmates', [])
        
        for player in all_players:
            if player.lower() in text.lower():
                return player
        return None
    
    def create_training_examples(self) -> List[Dict]:
        """Create training examples from games."""
        examples = []
        
        for game in self.games:
            game_context = self._build_game_context(game)
            
            for i, message in enumerate(game['messages']):
                annotation = self.annotate_deception(message, game)
                
                # Build input text
                if self.include_game_context:
                    input_text = f"{game_context}\n{message['speaker']}: {message['text']}"
                else:
                    input_text = f"{message['speaker']}: {message['text']}"
                
                # Create example
                example = {
                    'text': input_text,
                    'game_id': game['game_id'],
                    'speaker': message['speaker'],
                    'actual_role': annotation.actual_role or message.get('role', 'unknown'),
                    'is_deceptive': annotation.is_deceptive,
                    'deception_type': annotation.deception_type,
                    'message_idx': i,
                }
                
                examples.append(example)
        
        self.processed_data = examples
        return examples
    
    def _build_game_context(self, game: Dict) -> str:
        """Build context string for the game."""
        num_impostors = len(game.get('impostors', []))
        num_crewmates = len(game.get('crewmates', []))
        
        context = f"[Among Us Emergency Meeting]\n"
        context += f"Players: {num_impostors + num_crewmates} total "
        context += f"({num_impostors} impostors, {num_crewmates} crewmates)\n"
        
        return context
    
    def prepare_for_training(self) -> DatasetDict:
        """Prepare dataset for model training."""
        if not self.processed_data:
            self.create_training_examples()
        
        # Tokenize examples
        tokenized_examples = []
        for example in self.processed_data:
            tokens = self.tokenizer(
                example['text'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            example.update({
                'input_ids': tokens['input_ids'].squeeze(),
                'attention_mask': tokens['attention_mask'].squeeze(),
                'labels': tokens['input_ids'].squeeze(),  # For language modeling
                'deception_labels': int(example['is_deceptive']),
                'role_labels': 1 if example['actual_role'] == 'impostor' else 0
            })
            
            tokenized_examples.append(example)
        
        # Split into train/val/test
        dataset = Dataset.from_list(tokenized_examples)
        
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, len(dataset)))
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })


class RealDiplomacyDataset:
    """Loader for Diplomacy game data with betrayal annotations.
    
    Features:
    - Private messages between players
    - Public press statements  
    - Actual moves vs promised moves
    - Betrayal detection labels
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.games = []
        
    def load_data(self):
        """Load Diplomacy dataset."""
        diplomacy_file = self.data_path / "diplomacy_games.json"
        
        if diplomacy_file.exists():
            with open(diplomacy_file) as f:
                self.games = json.load(f)
        else:
            raise FileNotFoundError(f"No Diplomacy data found at {diplomacy_file}")
    
    def annotate_betrayal(self, message: Dict) -> DeceptionAnnotation:
        """Annotate messages with betrayal/deception labels."""
        is_betrayal = message.get('is_betrayal', False)
        
        if is_betrayal:
            # Analyze the type of betrayal
            text = message['text'].lower()
            actual_move = message.get('actual_move', '').lower()
            
            deception_type = 'coalition_betrayal'  # default
            
            if 'support' in text and 'support' not in actual_move:
                deception_type = 'broken_support'
            elif 'not' in text and 'move' in text:
                deception_type = 'false_promise'
            elif 'alliance' in text or 'together' in text:
                deception_type = 'alliance_betrayal'
            
            return DeceptionAnnotation(
                is_deceptive=True,
                deception_type=deception_type,
                confidence=0.9
            )
        
        return DeceptionAnnotation(is_deceptive=False)
    
    def create_training_examples(self) -> List[Dict]:
        """Create training examples from Diplomacy games."""
        examples = []
        
        for game in self.games:
            for message in game.get('messages', []):
                annotation = self.annotate_betrayal(message)
                
                # Build input with context
                input_text = f"[Diplomacy {message.get('season', 'Unknown')}]\n"
                input_text += f"{message['from']} to {message['to']}: {message['text']}"
                
                if message.get('actual_move'):
                    input_text += f"\n[Actual: {message['actual_move']}]"
                
                example = {
                    'text': input_text,
                    'game_id': game['game_id'],
                    'from_player': message['from'],
                    'to_player': message['to'],
                    'season': message.get('season', 'unknown'),
                    'is_deceptive': annotation.is_deceptive,
                    'deception_type': annotation.deception_type,
                    'is_betrayal': message.get('is_betrayal', False)
                }
                
                examples.append(example)
        
        return examples


class UnifiedDeceptionDataset:
    """Unified dataset combining multiple social deduction games."""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.datasets = {}
        self.all_examples = []
        
    def add_dataset(self, name: str, dataset):
        """Add a dataset to the unified collection."""
        self.datasets[name] = dataset
        
    def load_all_datasets(self, data_dir: str):
        """Load all available datasets."""
        data_path = Path(data_dir)
        
        # Try to load Among Us
        if (data_path / "among_us").exists():
            among_us = RealAmongUsDataset(
                data_path / "among_us",
                self.tokenizer,
                self.max_length
            )
            among_us.load_data()
            self.add_dataset('among_us', among_us)
        
        # Try to load Diplomacy
        if (data_path / "diplomacy").exists():
            diplomacy = RealDiplomacyDataset(
                data_path / "diplomacy",
                self.tokenizer,
                self.max_length
            )
            diplomacy.load_data()
            self.add_dataset('diplomacy', diplomacy)
    
    def create_unified_training_set(self) -> DatasetDict:
        """Create unified training set from all datasets."""
        all_examples = []
        
        for name, dataset in self.datasets.items():
            examples = dataset.create_training_examples()
            
            # Add dataset source
            for ex in examples:
                ex['dataset_source'] = name
            
            all_examples.extend(examples)
        
        # Shuffle and tokenize
        np.random.shuffle(all_examples)
        
        # Tokenize
        tokenized = []
        for ex in all_examples:
            tokens = self.tokenizer(
                ex['text'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            ex.update({
                'input_ids': tokens['input_ids'].squeeze(),
                'attention_mask': tokens['attention_mask'].squeeze(),
                'deception_labels': int(ex.get('is_deceptive', False))
            })
            
            tokenized.append(ex)
        
        # Create dataset splits
        dataset = Dataset.from_list(tokenized)
        
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        
        return DatasetDict({
            'train': dataset.select(range(train_size)),
            'validation': dataset.select(range(train_size, train_size + val_size)),
            'test': dataset.select(range(train_size + val_size, len(dataset)))
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'total_examples': len(self.all_examples),
            'datasets': {},
            'deception_distribution': {}
        }
        
        for name, dataset in self.datasets.items():
            examples = dataset.create_training_examples()
            deceptive_count = sum(1 for ex in examples if ex.get('is_deceptive', False))
            
            stats['datasets'][name] = {
                'num_examples': len(examples),
                'num_deceptive': deceptive_count,
                'deception_rate': deceptive_count / len(examples) if examples else 0
            }
        
        return stats