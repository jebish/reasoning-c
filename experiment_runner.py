"""
Experiment Runner
Main orchestrator for running reasoning experiments
"""

import os
import time
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
from tqdm import tqdm

from providers import ProviderManager
from reasoning_approaches import ReasoningApproachManager
from dataset_loader import DatasetLoader


class ExperimentRunner:
    """Main experiment runner for reasoning approaches"""
    
    def __init__(self, 
                 provider_manager: ProviderManager,
                 reasoning_manager: ReasoningApproachManager,
                 dataset_loader: DatasetLoader):
        self.provider_manager = provider_manager
        self.reasoning_manager = reasoning_manager
        self.dataset_loader = dataset_loader
        self.results = []
    
    def run_single_experiment(self, 
                            input_text: str,
                            reasoning_approach: str,
                            model: str,
                            **kwargs) -> Dict[str, Any]:
        """Run a single experiment"""
        try:
            # Execute reasoning approach
            result = self.reasoning_manager.execute_reasoning(
                reasoning_approach, 
                input_text, 
                self.provider_manager, 
                model, 
                **kwargs
            )
            
            # Get provider info
            provider_info = self.provider_manager.get_provider_info()
            current_provider = provider_info["current_provider"]
            
            # Format result
            experiment_result = {
                "input": input_text,
                "model_output": result.final_answer,
                "reasoning_trace": result.reasoning_trace,
                "execution_time_s": result.execution_time,
                "reasoning_approach": result.approach_name,
                "provider": current_provider,
                "model_name": model,
                "timestamp": datetime.now().isoformat(),
                "metadata": result.metadata or {}
            }
            
            return experiment_result
            
        except Exception as e:
            return {
                "input": input_text,
                "model_output": f"Error: {str(e)}",
                "reasoning_trace": f"Error occurred: {str(e)}",
                "execution_time_s": 0,
                "reasoning_approach": reasoning_approach,
                "provider": self.provider_manager.current_provider,
                "model_name": model,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def run_batch_experiments(self, 
                            dataset: pd.DataFrame,
                            reasoning_approaches: List[str],
                            models: List[str],
                            max_samples: Optional[int] = None,
                            save_interval: int = 10,
                            output_dir: str = "results") -> pd.DataFrame:
        """
        Run batch experiments on a dataset
        
        Args:
            dataset: Input dataset with 'input' column
            reasoning_approaches: List of reasoning approaches to test
            models: List of models to test
            max_samples: Maximum number of samples to process
            save_interval: Save results every N experiments
            output_dir: Directory to save results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Limit dataset if specified
        if max_samples and max_samples < len(dataset):
            dataset = dataset.head(max_samples)
        
        # Calculate total experiments
        total_experiments = len(dataset) * len(reasoning_approaches) * len(models)
        
        print(f"Starting batch experiments:")
        print(f"- Dataset samples: {len(dataset)}")
        print(f"- Reasoning approaches: {len(reasoning_approaches)}")
        print(f"- Models: {len(models)}")
        print(f"- Total experiments: {total_experiments}")
        
        # Initialize results
        self.results = []
        experiment_count = 0
        
        # Progress bar
        pbar = tqdm(total=total_experiments, desc="Running experiments")
        
        try:
            for _, row in dataset.iterrows():
                input_text = row["input"]
                
                for reasoning_approach in reasoning_approaches:
                    for model in models:
                        # Set current provider based on model
                        self._set_provider_for_model(model)
                        
                        # Run experiment
                        result = self.run_single_experiment(
                            input_text, 
                            reasoning_approach, 
                            model
                        )
                        
                        self.results.append(result)
                        experiment_count += 1
                        pbar.update(1)
                        
                        # Save intermediate results
                        if experiment_count % save_interval == 0:
                            self._save_intermediate_results(output_dir, experiment_count)
                        
                        # Small delay to avoid rate limiting
                        time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\nExperiment interrupted by user")
        finally:
            pbar.close()
        
        # Save final results
        return self._save_final_results(output_dir)
    
    def _set_provider_for_model(self, model: str):
        """Set the appropriate provider based on model name"""
        if "claude" in model.lower():
            if "anthropic" in self.provider_manager.providers:
                self.provider_manager.set_current_provider("anthropic")
        elif "gpt" in model.lower() or "openai" in model.lower():
            if "openrouter" in self.provider_manager.providers:
                self.provider_manager.set_current_provider("openrouter")
        elif "command" in model.lower():
            if "cohere" in self.provider_manager.providers:
                self.provider_manager.set_current_provider("cohere")
        else:
            # Default to first available provider
            if self.provider_manager.providers:
                first_provider = list(self.provider_manager.providers.keys())[0]
                self.provider_manager.set_current_provider(first_provider)
    
    def _save_intermediate_results(self, output_dir: str, experiment_count: int):
        """Save intermediate results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intermediate_results_{experiment_count}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f"Saved intermediate results: {filepath}")
    
    def _save_final_results(self, output_dir: str) -> pd.DataFrame:
        """Save final results and return DataFrame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_results_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        
        # Also save as JSON for metadata preservation
        json_filename = f"experiment_results_{timestamp}.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        with open(json_filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Saved final results:")
        print(f"- CSV: {filepath}")
        print(f"- JSON: {json_filepath}")
        
        return df
    
    def push_to_huggingface(self, 
                           df: pd.DataFrame, 
                           repo_name: str, 
                           hf_token: str,
                           commit_message: str = "Add experiment results") -> bool:
        """
        Push results to HuggingFace Hub
        
        Args:
            df: Results DataFrame
            repo_name: HuggingFace repository name
            hf_token: HuggingFace token
            commit_message: Commit message
        """
        try:
            from huggingface_hub import HfApi, create_repo
            
            api = HfApi(token=hf_token)
            
            # Create repo if it doesn't exist
            try:
                create_repo(repo_id=repo_name, token=hf_token, exist_ok=True)
            except Exception as e:
                print(f"Note: Repository creation issue (may already exist): {e}")
            
            # Save results to temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = f"experiment_results_{timestamp}.csv"
            temp_filepath = f"/tmp/{temp_filename}"
            
            df.to_csv(temp_filepath, index=False)
            
            # Upload to HuggingFace
            api.upload_file(
                path_or_fileobj=temp_filepath,
                path_in_repo=temp_filename,
                repo_id=repo_name,
                commit_message=commit_message
            )
            
            # Clean up temporary file
            os.remove(temp_filepath)
            
            print(f"Successfully pushed results to: https://huggingface.co/datasets/{repo_name}")
            return True
            
        except Exception as e:
            print(f"Error pushing to HuggingFace: {str(e)}")
            return False
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of experiment results"""
        if not self.results:
            return {"message": "No experiments run yet"}
        
        df = pd.DataFrame(self.results)
        
        summary = {
            "total_experiments": len(df),
            "unique_approaches": df["reasoning_approach"].nunique(),
            "unique_models": df["model_name"].nunique(),
            "unique_providers": df["provider"].nunique(),
            "average_execution_time": df["execution_time_s"].mean(),
            "total_execution_time": df["execution_time_s"].sum(),
            "error_rate": (df["model_output"].str.contains("Error", na=False).sum() / len(df)) * 100,
            "approaches_used": df["reasoning_approach"].unique().tolist(),
            "models_used": df["model_name"].unique().tolist(),
            "providers_used": df["provider"].unique().tolist()
        }
        
        return summary
    
    def compare_approaches(self) -> pd.DataFrame:
        """Compare performance across different reasoning approaches"""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        
        comparison = df.groupby("reasoning_approach").agg({
            "execution_time_s": ["mean", "std", "min", "max"],
            "model_output": lambda x: (x.str.contains("Error", na=False).sum() / len(x)) * 100
        }).round(3)
        
        comparison.columns = ["avg_time", "std_time", "min_time", "max_time", "error_rate"]
        comparison = comparison.sort_values("avg_time")
        
        return comparison
    
    def compare_models(self) -> pd.DataFrame:
        """Compare performance across different models"""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        
        comparison = df.groupby("model_name").agg({
            "execution_time_s": ["mean", "std", "min", "max"],
            "model_output": lambda x: (x.str.contains("Error", na=False).sum() / len(x)) * 100
        }).round(3)
        
        comparison.columns = ["avg_time", "std_time", "min_time", "max_time", "error_rate"]
        comparison = comparison.sort_values("avg_time")
        
        return comparison

