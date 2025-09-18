import time
import re
from typing import Optional
from collections import Counter
from .base import BaseReasoningApproach, ReasoningResult 
from .simple_approaches import ChainOfThoughtApproach

class ChainOfThoughtSCApproach(BaseReasoningApproach):
    """
    Self-Consistency with Chain-of-Thought (CoT-SC).
    This approach generates multiple diverse reasoning paths and selects the most
    consistent final answer through a majority vote.

    Source: Wang et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (2022)
    https://arxiv.org/abs/2203.11171
    """
    
    def __init__(self, num_paths: int = 5):
        """
        Initializes the CoT-SC approach.
        
        Args:
            num_paths (int): The number of diverse reasoning paths to generate. 
                             Must be at least 2. Defaults to 5.
        """
        super().__init__("Chain-of-Thought (CoT-SC)")
        if num_paths < 2:
            raise ValueError("num_paths for Self-Consistency must be at least 2.")
        self.num_paths = num_paths
        # This approach uses a standard CoT generator as its base
        self.base_cot_approach = ChainOfThoughtApproach()
    
    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        start_time = time.time()
        
        reasoning_paths = []
        final_answers = []
        
        # --- 1. SAMPLING: Generate multiple diverse reasoning paths ---
        # The key is to use a temperature > 0 to get diverse outputs.
        # We make a copy of kwargs to avoid modifying the original dict.
        sampling_kwargs = kwargs.copy()
        if 'temperature' not in sampling_kwargs:
            sampling_kwargs['temperature'] = 0.7  # A reasonable default for diversity
            
        for i in range(self.num_paths):
            # Call the base CoT reasoner to generate one path
            path_result = self.base_cot_approach.reason(input_text, provider_manager, model, **sampling_kwargs)
            
            # The 'final_answer' from the base CoT is the full reasoning chain
            full_reasoning_path = path_result.reasoning_trace
            reasoning_paths.append(f"--- Path {i+1} ---\n{full_reasoning_path}")
            
            # Extract just the final answer from the path for voting
            extracted_answer = self._extract_final_answer(full_reasoning_path)
            if extracted_answer:
                final_answers.append(extracted_answer)

        # --- 2. AGGREGATION: Perform a majority vote ---
        if not final_answers:
            final_answer = "Could not determine a consistent answer from any of the paths."
            most_common_answer_info = "No valid answers were extracted to vote on."
        else:
            vote_counts = Counter(final_answers)
            # Find the most common answer and its frequency
            most_common_tuple = vote_counts.most_common(1)[0]
            most_common_answer = most_common_tuple[0]
            
            final_answer = most_common_answer
            most_common_answer_info = dict(vote_counts)

        execution_time = time.time() - start_time
        
        # The reasoning_trace for CoT-SC is a concatenation of ALL generated paths
        full_reasoning_trace = "\n\n".join(reasoning_paths)
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_trace=full_reasoning_trace,
            execution_time=execution_time,
            approach_name=self.name,
            metadata={
                "num_paths_generated": self.num_paths,
                "all_extracted_answers": final_answers,
                "vote_counts": most_common_answer_info,
                "winning_answer": final_answer,
                "citation": "Wang et al. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv:2203.11171"
            }
        )

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """
        Extracts the final answer from a block of text using a series of patterns.
        This is a critical step for CoT-SC's success.
        """
        # It's better to extract from the <final> tag if it exists, for compatibility
        if "<final>" in text:
            match = re.search(r"<final>(.*?)</final>", text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # General patterns for multiple-choice (A-E) and numerical answers
        patterns = [
            r"(?:the final answer is|the answer is|final answer:|is therefore)\s*:?\s*([A-E](?=\)|\.|$)|-?\d+(?:\.\d+)?)",
            r'Answer:\s*([A-E](?=\)|\.|$)|-?\d+(?:\.\d+)?)',
            r'\b([A-E])\s*(?:\)|\.)',  # For standalone options like "A)" or "B."
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper().strip()
        
        # Fallback: find the last number in the entire text block
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
            
        return None
    
