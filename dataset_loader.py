"""
Dataset Loading and Management
Handles loading datasets from HuggingFace and other sources
"""

import os
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datasets import load_dataset, Dataset
from huggingface_hub import login
import json


class DatasetLoader:
    """Handles dataset loading and preprocessing"""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token
        if hf_token:
            login(token=hf_token)
    
    def load_huggingface_dataset(self, 
                                dataset_name: str, 
                                split: str = "test", 
                                subset: Optional[str] = None,
                                num_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Load a dataset from HuggingFace
        
        Args:
            dataset_name: Name of the dataset (e.g., "gsm8k", "hellaswag")
            split: Dataset split to load (train, test, validation)
            subset: Optional subset of the dataset
            num_samples: Number of samples to load (None for all)
        """
        try:
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            # Convert to pandas DataFrame
            df = dataset.to_pandas()
            
            # Limit samples if specified
            if num_samples and num_samples < len(df):
                df = df.head(num_samples)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading dataset {dataset_name}: {str(e)}")
    
    def load_custom_dataset(self, file_path: str, file_type: str = "auto") -> pd.DataFrame:
        """
        Load a custom dataset from file
        
        Args:
            file_path: Path to the dataset file
            file_type: Type of file (csv, json, jsonl, auto)
        """
        if file_type == "auto":
            file_type = file_path.split('.')[-1].lower()
        
        try:
            if file_type == "csv":
                return pd.read_csv(file_path)
            elif file_type == "json":
                return pd.read_json(file_path)
            elif file_type == "jsonl":
                return pd.read_json(file_path, lines=True)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            raise Exception(f"Error loading custom dataset from {file_path}: {str(e)}")
    
    def create_sample_dataset(self) -> pd.DataFrame:
        """Create a sample dataset for testing"""
        sample_data = [
            {
                "input": "What is 15 + 27?",
                "expected_output": "42"
            },
            {
                "input": "If a train travels 120 miles in 2 hours, what is its average speed?",
                "expected_output": "60 miles per hour"
            },
            {
                "input": "A store has 50 apples. They sell 15 apples in the morning and 20 apples in the afternoon. How many apples are left?",
                "expected_output": "15 apples"
            },
            {
                "input": "What is the capital of France?",
                "expected_output": "Paris"
            },
            {
                "input": "If you have 3 red balls and 4 blue balls, what is the probability of picking a red ball?",
                "expected_output": "3/7 or approximately 0.429"
            }
        ]
        
        return pd.DataFrame(sample_data)
    
    def get_popular_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get information about popular reasoning datasets"""
        return {
            "gsm8k": {
                "name": "GSM8K",
                "description": "Grade school math word problems",
                "fields": ["question", "answer"],
                "input_field": "question",
                "expected_field": "answer"
            },
            "hellaswag": {
                "name": "HellaSwag",
                "description": "Commonsense reasoning dataset",
                "fields": ["ctx", "endings", "label"],
                "input_field": "ctx",
                "expected_field": "label"
            },
            "arc": {
                "name": "ARC",
                "description": "AI2 Reasoning Challenge",
                "fields": ["question", "choices", "answerKey"],
                "input_field": "question",
                "expected_field": "answerKey"
            },
            "mmlu": {
                "name": "MMLU",
                "description": "Massive Multitask Language Understanding",
                "fields": ["question", "choices", "answer"],
                "input_field": "question",
                "expected_field": "answer"
            },
            "strategyqa": {
                "name": "StrategyQA",
                "description": "Multi-hop reasoning questions",
                "fields": ["question", "answer"],
                "input_field": "question",
                "expected_field": "answer"
            },
            "commonsenseqa": {
                "name": "CommonsenseQA",
                "description": "Commonsense question answering",
                "fields": ["question", "choices", "answerKey"],
                "input_field": "question",
                "expected_field": "answerKey"
            }
        }
    
    def preprocess_dataset(self, df: pd.DataFrame, 
                          input_field: str = "input", 
                          expected_field: Optional[str] = None) -> pd.DataFrame:
        """
        Preprocess dataset for experimentation
        
        Args:
            df: Input DataFrame
            input_field: Name of the input field
            expected_field: Name of the expected output field (optional)
        """
        # Ensure input field exists
        if input_field not in df.columns:
            raise ValueError(f"Input field '{input_field}' not found in dataset columns: {list(df.columns)}")
        
        # Create standardized format
        processed_df = pd.DataFrame()
        processed_df["input"] = df[input_field]
        
        if expected_field and expected_field in df.columns:
            processed_df["expected_output"] = df[expected_field]
        else:
            processed_df["expected_output"] = None
        
        # Add metadata
        processed_df["dataset_index"] = range(len(processed_df))
        
        return processed_df
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataset format and content"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # Check required columns
        required_columns = ["input"]
        for col in required_columns:
            if col not in df.columns:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Missing required column: {col}")
        
        if not validation_results["valid"]:
            return validation_results
        
        # Check for empty inputs
        empty_inputs = df["input"].isna().sum()
        if empty_inputs > 0:
            validation_results["warnings"].append(f"Found {empty_inputs} empty inputs")
        
        # Basic statistics
        validation_results["stats"] = {
            "total_samples": len(df),
            "empty_inputs": empty_inputs,
            "columns": list(df.columns),
            "sample_input": df["input"].iloc[0] if len(df) > 0 else None
        }
        
        return validation_results
    
    def save_dataset(self, df: pd.DataFrame, file_path: str, file_type: str = "auto"):
        """Save dataset to file"""
        if file_type == "auto":
            file_type = file_path.split('.')[-1].lower()
        
        try:
            if file_type == "csv":
                df.to_csv(file_path, index=False)
            elif file_type == "json":
                df.to_json(file_path, orient="records", indent=2)
            elif file_type == "jsonl":
                df.to_json(file_path, orient="records", lines=True)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            raise Exception(f"Error saving dataset to {file_path}: {str(e)}")
    
    def load_from_url(self, url: str, file_type: str = "auto") -> pd.DataFrame:
        """Load dataset from URL"""
        import requests
        import io
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            if file_type == "auto":
                file_type = url.split('.')[-1].lower()
            
            if file_type == "csv":
                return pd.read_csv(io.StringIO(response.text))
            elif file_type == "json":
                return pd.read_json(io.StringIO(response.text))
            elif file_type == "jsonl":
                return pd.read_json(io.StringIO(response.text), lines=True)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            raise Exception(f"Error loading dataset from URL {url}: {str(e)}")

