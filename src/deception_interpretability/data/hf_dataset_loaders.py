"""HuggingFace dataset loaders for social deduction games."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm


class SocialMazeDataset:
    """Loader for MBZUAI/SocialMaze dataset.
    
    A comprehensive multi-modal dataset for social deduction games
    including Among Us, Werewolf, and social reasoning tasks.
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        game_filter: Optional[str] = None  # 'amongus', 'werewolf', etc.
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.game_filter = game_filter
        self.dataset = None
        self.processed_data = []
        
    def load_from_huggingface(self):
        """Load dataset directly from HuggingFace."""
        print("Loading MBZUAI/SocialMaze from HuggingFace...")
        self.dataset = load_dataset("MBZUAI/SocialMaze")
        print(f"Available splits in SocialMaze: {list(self.dataset.keys())}")
        return self.dataset
    
    def process_for_training(self) -> DatasetDict:
        """Process dataset for model training."""
        if self.dataset is None:
            self.load_from_huggingface()
        
        processed_splits = {}
        
        # Map SocialMaze splits to standard names
        split_mapping = {
            'easy': 'train',
            'hard': 'validation'
        }
        
        for original_split, new_split in split_mapping.items():
            if original_split in self.dataset:
                print(f"Processing {original_split} split as {new_split}...")
                examples = []
                
                # Use streaming to avoid memory issues
                for item in tqdm(self.dataset[original_split]):
                    # Process item based on structure
                    processed = self._process_item(item)
                    if processed:
                        examples.extend(processed)
                    
                    # Stop after reasonable amount for training
                    if len(examples) >= 10000:
                        break
                
                # Tokenize examples
                tokenized_examples = self._tokenize_examples(examples)
                
                # Create dataset
                if tokenized_examples:
                    processed_splits[new_split] = Dataset.from_list(tokenized_examples)
        
        # Create test split from validation
        if 'validation' in processed_splits and len(processed_splits['validation']) > 1000:
            val_data = processed_splits['validation']
            split_point = int(len(val_data) * 0.5)
            processed_splits['validation'] = val_data.select(range(split_point))
            processed_splits['test'] = val_data.select(range(split_point, len(val_data)))
        
        return DatasetDict(processed_splits)
    
    def _process_item(self, item: Dict) -> List[Dict]:
        """Process a single item from SocialMaze."""
        examples = []
        
        # SocialMaze has specific structure with rounds
        task = item.get('task', '')
        system_prompt = item.get('system_prompt', '')
        prompt = item.get('prompt', '')
        answer = item.get('answer', '')
        reasoning = item.get('reasoning_process', '')
        
        # Extract role from answer (e.g., "My Role Is Lunatic")
        role = 'unknown'
        if 'My Role Is' in answer:
            role = answer.split('My Role Is')[1].strip().split('.')[0].lower()
        
        # Process each round
        for round_key in ['round 1', 'round 2', 'round 3']:
            if round_key in item:
                round_text = item[round_key]
                
                # Parse statements from round
                statements = round_text.split('\n')
                for statement in statements:
                    if 'says' in statement:
                        # Extract player and statement
                        parts = statement.split('says', 1)
                        if len(parts) == 2:
                            speaker = parts[0].strip()
                            claim = parts[1].strip()
                            
                            # Determine if deceptive based on role
                            is_deceptive = self._detect_deception_socialmaze(claim, role, speaker)
                            
                            # Create training example
                            input_text = f"{system_prompt[:200]}\n{round_key}: {speaker} says {claim}"
                            
                            example = {
                                'text': input_text,
                                'speaker': speaker,
                                'claim': claim,
                                'round': round_key,
                                'role': role,
                                'is_deceptive': is_deceptive,
                                'game_type': 'social_deduction'
                            }
                            examples.append(example)
        
        # Also create an example from the full game
        if answer and reasoning:
            full_text = f"{system_prompt[:200]}\n{prompt[:100]}\nReasoning: {reasoning[:200]}"
            
            # Criminals and Lunatics are deceptive roles
            is_deceptive = role in ['criminal', 'lunatic', 'rumormonger']
            
            example = {
                'text': full_text,
                'speaker': 'self',
                'claim': answer,
                'round': 'final',
                'role': role,
                'is_deceptive': is_deceptive,
                'game_type': 'social_deduction'
            }
            examples.append(example)
        
        return examples
    
    def _detect_deception_socialmaze(self, claim: str, role: str, speaker: str) -> bool:
        """Detect deception in SocialMaze based on role and claim."""
        claim_lower = claim.lower()
        
        # Criminals lie about who is criminal
        if role == 'criminal':
            if 'is the criminal' in claim_lower or 'is not the criminal' in claim_lower:
                return True
                
        # Lunatics think they're criminals but aren't
        elif role == 'lunatic':
            if 'criminal' in claim_lower:
                return True
                
        # Rumormongers may provide incorrect info
        elif role == 'rumormonger':
            return True  # They often give wrong info
            
        return False
    
    def _detect_deception(self, message: Any, role: Optional[str]) -> bool:
        """Detect if a message is deceptive based on content and role."""
        if isinstance(message, dict):
            # Check for explicit deception labels
            if 'is_deceptive' in message:
                return message['is_deceptive']
            if 'is_lie' in message:
                return message['is_lie']
            
            text = message.get('text', '').lower()
        else:
            text = str(message).lower()
        
        # Role-based deception detection
        if role:
            role_lower = role.lower()
            if any(evil in role_lower for evil in ['impostor', 'werewolf', 'mafia', 'traitor']):
                # Evil roles claiming innocence
                innocent_claims = ['innocent', 'not the', 'trust me', 'crewmate', 'villager']
                if any(claim in text for claim in innocent_claims):
                    return True
        
        return False
    
    def _tokenize_examples(self, examples: List[Dict]) -> List[Dict]:
        """Tokenize examples for training."""
        tokenized = []
        
        for ex in examples:
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
                'labels': tokens['input_ids'].squeeze(),
                'deception_labels': int(ex.get('is_deceptive', False))
            })
            
            # Add role labels if available
            role = ex.get('role', 'unknown').lower()
            if 'impostor' in role or 'werewolf' in role:
                ex['role_labels'] = 1  # Evil
            elif 'crewmate' in role or 'villager' in role:
                ex['role_labels'] = 0  # Good
            else:
                ex['role_labels'] = 2  # Unknown
            
            tokenized.append(ex)
        
        return tokenized


class WerewolfAmongUsDataset:
    """Loader for bolinlai/Werewolf-Among-Us dataset.

    One Night Ultimate Werewolf games from Ego4D and Youtube sources.
    Each game has Dialogue with strategy annotations and player roles.

    Note: load_dataset() fails on this repo due to mixed types in
    votingOutcome ('NA' strings vs ints). We load the JSON files directly.
    """

    # Roles that are deceptive (evil team or want to appear evil)
    EVIL_ROLES = {'werewolf', 'minion'}
    # Tanner wants to get voted out (so they *want* to seem suspicious — different kind of deception)
    TRICKY_ROLES = {'tanner'}

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        game_type: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.game_type = game_type
        self.raw_games: Dict[str, List[Dict]] = {}  # split_name -> list of games

    def load_from_huggingface(self):
        """Load JSON files directly from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download

        print("Loading bolinlai/Werewolf-Among-Us from HuggingFace...")

        sources = ['Ego4D', 'Youtube']
        split_files = {'train': 'train.json', 'validation': 'val.json', 'test': 'test.json'}

        for source in sources:
            for split_name, filename in split_files.items():
                repo_path = f"{source}/split/{filename}"
                try:
                    local_path = hf_hub_download(
                        'bolinlai/Werewolf-Among-Us', repo_path, repo_type='dataset'
                    )
                    with open(local_path) as f:
                        games = json.load(f)
                    # Tag each game with its source
                    for g in games:
                        g['_source'] = source
                    self.raw_games.setdefault(split_name, []).extend(games)
                    print(f"  Loaded {len(games)} games from {source}/{split_name}")
                except Exception as e:
                    print(f"  Skipping {repo_path}: {e}")

        total = sum(len(v) for v in self.raw_games.values())
        print(f"Total games loaded: {total} across splits {list(self.raw_games.keys())}")

    def process_for_training(self) -> DatasetDict:
        """Process dataset for training."""
        if not self.raw_games:
            self.load_from_huggingface()

        processed_splits = {}

        for split_name, games in self.raw_games.items():
            print(f"Processing {split_name} split ({len(games)} games)...")
            examples = []

            for game in tqdm(games):
                processed = self._process_game(game)
                examples.extend(processed)

            if examples:
                tokenized = self._tokenize_examples(examples)
                processed_splits[split_name] = Dataset.from_list(tokenized)
                print(f"  {split_name}: {len(tokenized)} examples")

        return DatasetDict(processed_splits)

    def _process_game(self, game: Dict) -> List[Dict]:
        """Process a single game into training examples."""
        examples = []

        player_names = game.get('playerNames', [])
        start_roles = game.get('startRoles', [])
        end_roles = game.get('endRoles', [])
        dialogue = game.get('Dialogue', [])
        game_id = game.get('Game_ID', game.get('YT_ID', ''))
        source = game.get('_source', 'unknown')

        # Build role mappings
        start_role_map = dict(zip(player_names, start_roles))
        end_role_map = dict(zip(player_names, end_roles))

        for i, entry in enumerate(dialogue):
            speaker = entry.get('speaker', 'unknown')
            utterance = entry.get('utterance', '')
            annotations = entry.get('annotation', [])

            if not utterance.strip():
                continue

            start_role = start_role_map.get(speaker, 'unknown')
            end_role = end_role_map.get(speaker, start_role)

            is_deceptive = self._is_deceptive(
                speaker, utterance, annotations, start_role, end_role
            )

            # Build input text with context
            input_text = f"[werewolf] {speaker} ({start_role}): {utterance}"

            examples.append({
                'text': input_text,
                'game_id': f"{source}_{game_id}",
                'game_type': 'werewolf',
                'speaker': speaker,
                'role': start_role,
                'end_role': end_role,
                'message_idx': i,
                'annotation': ','.join(annotations),
                'is_deceptive': is_deceptive,
            })

        return examples

    def _is_deceptive(
        self,
        speaker: str,
        utterance: str,
        annotations: List[str],
        start_role: str,
        end_role: str,
    ) -> bool:
        """Determine if an utterance is deceptive based on role and strategy."""
        role_lower = start_role.lower()

        # Evil roles using strategic speech acts are deceptive
        if role_lower in self.EVIL_ROLES:
            strategic_annotations = {
                'Identity Declaration', 'Defense', 'Accusation', 'Evidence'
            }
            if any(a in strategic_annotations for a in annotations):
                return True

        # Tanner trying to get voted out via misdirection
        if role_lower in self.TRICKY_ROLES:
            if any(a in {'Accusation', 'Identity Declaration'} for a in annotations):
                return True

        return False
    
    def _tokenize_examples(self, examples: List[Dict]) -> List[Dict]:
        """Tokenize examples."""
        tokenized = []
        
        for ex in examples:
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
                'labels': tokens['input_ids'].squeeze(),
                'deception_labels': int(ex['is_deceptive'])
            })
            
            # Role labels
            role = ex.get('role', '').lower()
            if any(evil in role for evil in ['impostor', 'werewolf', 'mafia', 'minion']):
                ex['role_labels'] = 1  # Evil
            elif any(good in role for good in [
                'crewmate', 'villager', 'innocent', 'seer', 'robber',
                'troublemaker', 'insomniac',
            ]):
                ex['role_labels'] = 0  # Good
            else:
                ex['role_labels'] = 2  # Other (tanner, etc.)

            tokenized.append(ex)

        return tokenized


class UnifiedHFDataset:
    """Unified loader for all HuggingFace social deduction datasets."""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        datasets_to_load: List[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.datasets_to_load = datasets_to_load or ['socialmaze', 'werewolf-amongus']
        self.datasets = {}
        
    def load_all_datasets(self) -> DatasetDict:
        """Load and combine all specified datasets."""
        all_splits = {}
        
        # Load SocialMaze
        if 'socialmaze' in self.datasets_to_load:
            try:
                print(f"Loading SocialMaze... (datasets_to_load: {self.datasets_to_load})")
                socialmaze = SocialMazeDataset(self.tokenizer, self.max_length)
                socialmaze_data = socialmaze.process_for_training()
                
                for split, data in socialmaze_data.items():
                    if split not in all_splits:
                        all_splits[split] = []
                    # Add source tag
                    data = data.map(lambda x: {'source': 'socialmaze', **x})
                    all_splits[split].append(data)
                print(f"SocialMaze loaded successfully")
                    
            except Exception as e:
                print(f"Error loading SocialMaze: {e}")
        
        # Load Werewolf-AmongUs
        if 'werewolf-amongus' in self.datasets_to_load:
            try:
                print(f"Loading Werewolf-AmongUs... (datasets_to_load: {self.datasets_to_load})")
                werewolf_amongus = WerewolfAmongUsDataset(self.tokenizer, self.max_length)
                wa_data = werewolf_amongus.process_for_training()
                
                for split, data in wa_data.items():
                    if split not in all_splits:
                        all_splits[split] = []
                    # Add source tag
                    data = data.map(lambda x: {'source': 'werewolf-amongus', **x})
                    all_splits[split].append(data)
                print(f"Werewolf-AmongUs loaded successfully")
                    
            except Exception as e:
                print(f"Error loading Werewolf-AmongUs: {e}")
        
        # Combine datasets
        combined_splits = {}
        for split, datasets in all_splits.items():
            if datasets:
                # Concatenate all datasets for this split
                from datasets import concatenate_datasets
                combined_splits[split] = concatenate_datasets(datasets)
                
                # Shuffle
                combined_splits[split] = combined_splits[split].shuffle(seed=42)
                
                print(f"{split}: {len(combined_splits[split])} examples")
        
        return DatasetDict(combined_splits)
    
    def get_statistics(self, dataset_dict: DatasetDict) -> Dict:
        """Get statistics about the loaded datasets."""
        stats = {
            'total_examples': sum(len(d) for d in dataset_dict.values()),
            'splits': {name: len(d) for name, d in dataset_dict.items()},
            'sources': {},
            'deception_rate': {}
        }
        
        for split_name, dataset in dataset_dict.items():
            # Count by source
            if 'source' in dataset.column_names:
                source_counts = {}
                for item in dataset:
                    source = item['source']
                    source_counts[source] = source_counts.get(source, 0) + 1
                stats['sources'][split_name] = source_counts
            
            # Calculate deception rate
            if 'deception_labels' in dataset.column_names:
                deceptive_count = sum(item['deception_labels'] for item in dataset)
                stats['deception_rate'][split_name] = deceptive_count / len(dataset)
        
        return stats