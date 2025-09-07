"""
Corrected Reasoning Approaches Implementation
Based on original research papers with proper citations and implementations
"""

import time
import json
import random
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict, deque, Counter
import heapq 


@dataclass
class ReasoningResult:
    """Result of a reasoning approach"""
    final_answer: str
    reasoning_trace: str
    execution_time: float
    approach_name: str
    metadata: Dict[str, Any] = None


class BaseReasoningApproach(ABC):
    """Base class for all reasoning approaches"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        """Execute the reasoning approach"""
        pass


class NoneApproach(BaseReasoningApproach):
    """No reasoning approach - direct response baseline"""
    
    def __init__(self):
        super().__init__("None")
    
    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        start_time = time.time()
        none_prompt = f"""Respond exactly in this format: <final>[answer]</final>
        {input_text}

        Respond exactly in this format: <final>[answer]</final>"""
        response = provider_manager.generate_response(input_text, model, **kwargs)
        
        execution_time = time.time() - start_time
        
        return ReasoningResult(
            final_answer=response["content"],
            reasoning_trace="Direct response without explicit reasoning",
            execution_time=execution_time,
            approach_name=self.name,
            metadata={"provider_response": response}
        )


class ChainOfThoughtApproach(BaseReasoningApproach):
    """
    Chain-of-Thought (CoT) reasoning
    Source: Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)
    https://arxiv.org/abs/2201.11903
    """
    
    def __init__(self):
        super().__init__("Chain-of-Thought (CoT)")
    
    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        start_time = time.time()
        
        cot_prompt = f"""Please solve the following problem step by step. Think through your reasoning process and show your work.

        Problem: {input_text}

        Lets think step by step , respond exactly in this format: 
        <reasoning>[few steps of reasoning step-by-step]</reasoning>
        <final>[final answer only]</final>:"""
        
        response = provider_manager.generate_response(cot_prompt, model, **kwargs)
        
        execution_time = time.time() - start_time
        
        return ReasoningResult(
            final_answer=response["content"],
            reasoning_trace=response["content"],
            execution_time=execution_time,
            approach_name=self.name,
            metadata={
                "provider_response": response,
                "citation": "Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. arXiv:2201.11903"
            }
        )


class LeastToMostApproach(BaseReasoningApproach):
    """
    Least-to-Most Prompting (LtM) - Corrected sequential implementation with structured final output.
    This version saves the FULL step-by-step reasoning in the trace, while the final output is formatted.
    Source: Zhou et al. "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models" (2022)
    """
    
    def __init__(self):
        super().__init__("Least-to-Most Prompting (LtM)")

    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        start_time = time.time()
        
        # --- Step 1: DECOMPOSITION ---
        decompose_prompt = f"Break down the complex problem into a sequence of smaller sub-problems. List them in order.\n\nProblem: {input_text}\n\nSub-problems:"
        decomposition_response = provider_manager.generate_response(decompose_prompt, model, **kwargs)
        decomposition_text = decomposition_response["content"]
        sub_problems = [line.strip() for line in decomposition_text.strip().split('\n') if line.strip()]

        if not sub_problems:
            return self._fallback_reasoning(input_text, provider_manager, model, **kwargs)

        # --- Step 2: SEQUENTIAL SOLVING (in a loop) ---
        solved_context = ""
        formatted_output = "Failed to generate final formatted output."
        
        # **NEW:** This variable will store the complete, raw reasoning process.
        full_reasoning_trace = f"--- DECOMPOSITION ---\n{decomposition_text}\n\n--- SEQUENTIAL SOLVING ---\n"
        
        for i, sub_problem in enumerate(sub_problems):
            is_last_step = (i == len(sub_problems) - 1)
            
            if not is_last_step:
                # --- INTERMEDIATE STEP PROMPT ---
                prompt = f"""**ORIGINAL PROBLEM:** {input_text}
**PREVIOUSLY SOLVED STEPS:**
{solved_context if solved_context else "None."}
**CURRENT SUB-PROBLEM TO SOLVE:** "{sub_problem}"

**SOLUTION TO CURRENT SUB-PROBLEM:**"""
                
                sub_solution_response = provider_manager.generate_response(prompt, model, **kwargs)
                sub_solution = sub_solution_response["content"]
                
                # Update the context for the next iteration
                solved_context += f"Sub-problem: {sub_problem}\nSolution: {sub_solution}\n\n"
                
                # **MODIFIED:** Append the raw step-by-step process to our full trace.
                full_reasoning_trace += f"Step {i+1}: Solving '{sub_problem}'\nResult: {sub_solution}\n\n"

            else:
                # --- FINAL STEP AND FORMATTING PROMPT ---
                # **MODIFIED:** The final prompt now needs the full raw trace to be able to synthesize it.
                final_prompt_reasoning_block = full_reasoning_trace + f"Step {len(sub_problems)}: Solving '{sub_problem}'\n"
                
                prompt = f"""**ROLE:** You are a problem solver completing the final step of a multi-part problem.
**TASK:** First, solve the final sub-problem. Then, synthesize the answer in the final format.

**FULL REASONING CONTEXT:**
{final_prompt_reasoning_block}

**FINAL SUB-PROBLEM TO SOLVE:**
"{sub_problem}"

**INSTRUCTIONS:**
1. Solve the final sub-problem.
2. Provide only the final answer to the original problem inside the <final> tag.

**FORMATTED OUTPUT (in the specified format)**
<final>[final answer only]</final>
:"""
                
                final_response = provider_manager.generate_response(prompt, model, **kwargs)
                formatted_output = final_response["content"]
                
                # **MODIFIED:** Append the final formatted output to the trace for full transparency.
                full_reasoning_trace += f"\n--- FINAL FORMATTED OUTPUT ---\n{formatted_output}"

        execution_time = time.time() - start_time
        
        # The final_answer is still extracted from the clean, formatted output.
        final_answer = self._extract_from_tag(formatted_output, "final")
        
        return ReasoningResult(
            final_answer=final_answer,
            # **MODIFIED:** The trace is now the full, detailed, raw reasoning process.
            reasoning_trace=full_reasoning_trace,
            execution_time=execution_time,
            approach_name=self.name,
            metadata={
                "decomposition": decomposition_text,
                "sub_problems_solved": len(sub_problems),
                "final_formatted_output": formatted_output, # Also save the clean output in metadata
                "citation": "Zhou et al. (2022). Least-to-Most Prompting Enables Complex Reasoning in Large Language Models. arXiv:2205.10625"
            }
        )
        
    def _extract_from_tag(self, text: str, tag: str) -> str:
        """Extracts content from a specific tag, e.g., <final>...</final>."""
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        lines = text.strip().split('\n')
        return lines[-1].strip() if lines else "Extraction failed"

    def _fallback_reasoning(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        """A simple CoT fallback for when decomposition fails."""
        # This can remain as is, or you can use your formatted CoT class.
        # For simplicity, we'll keep the original fallback.
        start_time = time.time()
        cot_prompt = f"Let's think step by step to solve the following problem:\n{input_text}"
        response = provider_manager.generate_response(cot_prompt, model, **kwargs)
        execution_time = time.time() - start_time
        return ReasoningResult(
            final_answer=response["content"],
            reasoning_trace="LtM Fallback: Decomposition failed. Used basic Chain-of-Thought.",
            execution_time=execution_time,
            approach_name=self.name,
            metadata={"decomposition_failed": True}
        )

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

class ReasoningAsPlanningApproach(BaseReasoningApproach):
    """
    Reasoning-as-Planning (RAP)
    Source: Hao et al. "Reasoning with Language Model is Planning with World Model" (2023)
    https://arxiv.org/abs/2305.14992
    """
    
    def __init__(self):
        super().__init__("Reasoning-as-Planning (RAP)")
    
    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        start_time = time.time()
        
        # Step 1: Create a plan (Planning phase)
        plan_prompt = f"""Create a detailed plan to solve the following problem. Think of this as creating a roadmap with specific steps and actions.

Problem: {input_text}

Create a step-by-step plan:"""
        
        plan_response = provider_manager.generate_response(plan_prompt, model, **kwargs)
        
        # Step 2: Execute the plan (Execution phase)
        execute_prompt = f"""Now execute the following plan step by step to solve the problem:

Problem: {input_text}

Plan:
{plan_response['content']}

Execute each step of the plan and provide the final answer:"""
        
        execution_response = provider_manager.generate_response(execute_prompt, model, **kwargs)
        
        execution_time = time.time() - start_time
        
        reasoning_trace = f"Plan: {plan_response['content']}\n\nExecution: {execution_response['content']}"
        
        return ReasoningResult(
            final_answer=execution_response["content"],
            reasoning_trace=reasoning_trace,
            execution_time=execution_time,
            approach_name=self.name,
            metadata={
                "plan": plan_response,
                "execution": execution_response,
                "citation": "Hao et al. (2023). Reasoning with Language Model is Planning with World Model. arXiv:2305.14992"
            }
        )


class ChainOfVerificationApproach(BaseReasoningApproach):
    """
    Chain-of-Verification (CoVe)
    Source: Dhuliawala et al. "Chain-of-Verification Reduces Hallucination in Large Language Models" (2023)
    https://arxiv.org/abs/2309.11495
    """
    
    def __init__(self):
        super().__init__("Chain-of-Verification (CoVe)")
    
    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        start_time = time.time()
        
        # Step 1: Generate initial answer
        initial_prompt = f"""Answer the following question:

{input_text}"""
        
        initial_response = provider_manager.generate_response(initial_prompt, model, **kwargs)
        
        # Step 2: Generate verification questions
        verify_prompt = f"""Given this question and answer, generate verification questions to check the correctness:

Question: {input_text}
Answer: {initial_response['content']}

Generate 3-5 verification questions to check if this answer is correct:"""
        
        verify_response = provider_manager.generate_response(verify_prompt, model, **kwargs)
        
        # Step 3: Answer verification questions
        answer_verify_prompt = f"""Answer these verification questions:

{verify_response['content']}

Provide clear answers to each verification question:"""
        
        answer_verify_response = provider_manager.generate_response(answer_verify_prompt, model, **kwargs)
        
        # Step 4: Final verification and correction
        final_prompt = f"""Based on the verification, provide a final corrected answer:

Original Question: {input_text}
Initial Answer: {initial_response['content']}
Verification Questions: {verify_response['content']}
Verification Answers: {answer_verify_response['content']}

Provide the final verified and corrected answer:"""
        
        final_response = provider_manager.generate_response(final_prompt, model, **kwargs)
        
        execution_time = time.time() - start_time
        
        reasoning_trace = f"Initial Answer: {initial_response['content']}\n\nVerification Questions: {verify_response['content']}\n\nVerification Answers: {answer_verify_response['content']}\n\nFinal Answer: {final_response['content']}"
        
        return ReasoningResult(
            final_answer=final_response["content"],
            reasoning_trace=reasoning_trace,
            execution_time=execution_time,
            approach_name=self.name,
            metadata={
                "initial_answer": initial_response,
                "verification_questions": verify_response,
                "verification_answers": answer_verify_response,
                "final_answer": final_response,
                "citation": "Dhuliawala et al. (2023). Chain-of-Verification Reduces Hallucination in Large Language Models. arXiv:2309.11495"
            }
        )


# class SkeletonOfThoughtApproach(BaseReasoningApproach):
#     """
#     Skeleton-of-Thought (SoT)
#     Source: Ning et al. "Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding" (2023)
#     https://arxiv.org/abs/2307.15337
#     """
    
#     def __init__(self):
#         super().__init__("Skeleton-of-Thought (SoT)")
    
#     def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
#         start_time = time.time()
        
#         # Step 1: Create skeleton (Parallel skeleton generation)
#         skeleton_prompt = f"""Create a skeleton/outline for answering the following question. Think of the main points and structure without going into details:

# Question: {input_text}

# Create a skeleton outline with main points:"""
        
#         skeleton_response = provider_manager.generate_response(skeleton_prompt, model, **kwargs)
        
#         # Step 2: Fill in the skeleton (Parallel point expansion)
#         fill_prompt = f"""Now fill in the skeleton with detailed reasoning and provide the final answer:

# Question: {input_text}

# Skeleton:
# {skeleton_response['content']}

# Fill in the details and provide the complete answer:"""
        
#         fill_response = provider_manager.generate_response(fill_prompt, model, **kwargs)
        
#         execution_time = time.time() - start_time
        
#         reasoning_trace = f"Skeleton: {skeleton_response['content']}\n\nDetailed Answer: {fill_response['content']}"
        
#         return ReasoningResult(
#             final_answer=fill_response["content"],
#             reasoning_trace=reasoning_trace,
#             execution_time=execution_time,
#             approach_name=self.name,
#             metadata={
#                 "skeleton": skeleton_response,
#                 "detailed_answer": fill_response,
#                 "citation": "Ning et al. (2023). Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding. arXiv:2307.15337"
#             }
#         )

class SkeletonOfThoughtApproach(BaseReasoningApproach):
    """
    Skeleton-of-Thought (SoT) - Complete Parallel Expansion Implementation
    Source: Ning et al. "Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding" (2023)
    https://arxiv.org/abs/2307.15337
    
    Features:
    - True parallel expansion using asyncio
    - Robust skeleton parsing with multiple patterns
    - Quality validation and aggregation phase
    - Context window management
    - Comprehensive error handling
    """
    
    def __init__(self, max_skeleton_points: int = 8, enable_aggregation: bool = True):
        super().__init__("Skeleton-of-Thought (SoT)")
        self.max_skeleton_points = max_skeleton_points
        self.enable_aggregation = enable_aggregation
        
        # Multiple regex patterns for robust outline parsing
        self.outline_patterns = [
            r'^\s*(\d+)[\.\)]\s*(.*)',          # 1. or 1)
            r'^\s*[\-\*\+]\s*(.*)',             # - or * or +
            r'^\s*([A-Za-z])[\.\)]\s*(.*)',     # A. or a)
            r'^\s*([IVX]+)[\.\)]\s*(.*)',       # Roman numerals
            r'^\s*\(([\d\w]+)\)\s*(.*)',        # (1) or (a)
        ]

    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        """Main reasoning method with full SoT implementation"""
        start_time = time.time()
        
        try:
            # Check if provider_manager supports async operations
            if hasattr(provider_manager, 'generate_response_async'):
                return asyncio.run(self._async_reason(input_text, provider_manager, model, **kwargs))
            else:
                return self._sync_reason(input_text, provider_manager, model, **kwargs)
        except Exception as e:
            return self._fallback_reasoning(input_text, provider_manager, model, 
                                          error=str(e), **kwargs)

    async def _async_reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        """Asynchronous implementation with true parallel expansion"""
        start_time = time.time()
        
        # Step 1: Generate skeleton
        skeleton_text = await self._generate_skeleton_async(input_text, provider_manager, model, **kwargs)
        
        # Step 2: Parse skeleton into points
        outline_points = self._parse_skeleton(skeleton_text)
        
        if not outline_points:
            return await self._fallback_reasoning_async(input_text, provider_manager, model, 
                                                      error="Skeleton parsing failed", **kwargs)
        
        # Limit number of points to manage context and complexity
        if len(outline_points) > self.max_skeleton_points:
            outline_points = outline_points[:self.max_skeleton_points]
        
        # Step 3: Parallel expansion of all points
        full_reasoning_trace = f"--- SKELETON ({len(outline_points)} points) ---\n{skeleton_text}\n\n--- PARALLEL EXPANSION ---\n"
        
        # Create expansion tasks for parallel execution
        expansion_tasks = [
            self._expand_point_async(i, point, outline_points, input_text, 
                                   provider_manager, model, **kwargs)
            for i, point in enumerate(outline_points)
        ]
        
        # Execute all expansions in parallel
        expanded_results = await asyncio.gather(*expansion_tasks, return_exceptions=True)
        
        # Process results and handle any failures
        expanded_parts = []
        for i, result in enumerate(expanded_results):
            if isinstance(result, Exception):
                expanded_parts.append(f"[Error expanding point {i+1}: {str(result)}]")
                full_reasoning_trace += f"Point {i+1} ERROR: {str(result)}\n"
            else:
                expanded_parts.append(result)
                full_reasoning_trace += f"\n--- Expanding Point {i+1}: \"{point_text}\" ---\n{result}\n"
        
        # Step 4: Aggregation (optional)
        if self.enable_aggregation and len(expanded_parts) > 1:
            final_answer = await self._aggregate_parts_async(
                input_text, skeleton_text, expanded_parts, 
                provider_manager, model, **kwargs
            )
            full_reasoning_trace += f"\n--- AGGREGATION ---\n{final_answer}\n"
        else:
            final_answer = self._simple_join(expanded_parts)
        
        execution_time = time.time() - start_time
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_trace=full_reasoning_trace,
            execution_time=execution_time,
            approach_name=self.name,
            metadata={
                "skeleton": skeleton_text,
                "outline_points": outline_points,
                "num_points": len(outline_points),
                "parallel_execution": True,
                "aggregation_used": self.enable_aggregation,
                "citation": "Ning et al. (2023). Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding. arXiv:2307.15337"
            }
        )

    def _sync_reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        """Synchronous fallback implementation"""
        start_time = time.time()
        
        # Step 1: Generate skeleton
        skeleton_text = self._generate_skeleton_sync(input_text, provider_manager, model, **kwargs)
        
        # Step 2: Parse skeleton
        outline_points = self._parse_skeleton(skeleton_text)
        
        if not outline_points:
            return self._fallback_reasoning(input_text, provider_manager, model, 
                                          error="Skeleton parsing failed", **kwargs)
        
        # Limit points
        if len(outline_points) > self.max_skeleton_points:
            outline_points = outline_points[:self.max_skeleton_points]
        
        # Step 3: Sequential expansion (since no async support)
        expanded_parts = []
        full_reasoning_trace = f"--- SKELETON ({len(outline_points)} points) ---\n{skeleton_text}\n\n--- SEQUENTIAL EXPANSION ---\n"
        
        for i, point in enumerate(outline_points):
            try:
                expanded_part = self._expand_point_sync(i, point, outline_points, input_text, 
                                                      provider_manager, model, **kwargs)
                expanded_parts.append(expanded_part)
                full_reasoning_trace += f"\n--- Expanding Point {i+1}: \"{point_text}\" ---\n{expanded_part}\n"
            except Exception as e:
                expanded_part = f"[Error expanding point {i+1}: {str(e)}]"
                expanded_parts.append(expanded_part)
                full_reasoning_trace += f"Point {i+1} ERROR: {str(e)}\n"
        
        # Step 4: Simple aggregation or joining
        if self.enable_aggregation and len(expanded_parts) > 1:
            final_answer = self._aggregate_parts_sync(
                input_text, skeleton_text, expanded_parts, 
                provider_manager, model, **kwargs
            )
            full_reasoning_trace += f"\n--- AGGREGATION ---\n{final_answer}\n"
        else:
            final_answer = self._simple_join(expanded_parts)
        
        execution_time = time.time() - start_time
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_trace=full_reasoning_trace,
            execution_time=execution_time,
            approach_name=self.name,
            metadata={
                "skeleton": skeleton_text,
                "outline_points": outline_points,
                "num_points": len(outline_points),
                "parallel_execution": False,
                "aggregation_used": self.enable_aggregation,
                "citation": "Ning et al. (2023). Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding. arXiv:2307.15337"
            }
        )

    async def _generate_skeleton_async(self, input_text: str, provider_manager, model: str, **kwargs) -> str:
        """Generate skeleton outline asynchronously"""
        skeleton_prompt = f"""Create a concise, numbered list outline (skeleton) for a comprehensive answer to the following question. 

REQUIREMENTS:
- Use numbered format (1., 2., 3., etc.)
- Each point should cover a distinct aspect or topic
- Keep each point brief but descriptive
- Aim for 3-7 main points
- Focus on logical flow and completeness

Question: {input_text}

Numbered Skeleton Outline:"""
        
        response = await provider_manager.generate_response_async(skeleton_prompt, model, **kwargs)
        return response["content"].strip()

    def _generate_skeleton_sync(self, input_text: str, provider_manager, model: str, **kwargs) -> str:
        """Generate skeleton outline synchronously"""
        skeleton_prompt = f"""Create a concise, numbered list outline (skeleton) for a comprehensive answer to the following question. 

REQUIREMENTS:
- Use numbered format (1., 2., 3., etc.)
- Each point should cover a distinct aspect or topic
- Keep each point brief but descriptive
- Aim for 3-7 main points
- Focus on logical flow and completeness

Question: {input_text}

Numbered Skeleton Outline:"""
        
        response = provider_manager.generate_response(skeleton_prompt, model, **kwargs)
        return response["content"].strip()

    def _parse_skeleton(self, skeleton_text: str) -> List[str]:
        """Parse skeleton text into individual points using multiple patterns"""
        outline_points = []
        lines = skeleton_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try each pattern
            for pattern in self.outline_patterns:
                match = re.match(pattern, line)
                if match:
                    # Extract the content part (usually the last group)
                    content = match.groups()[-1].strip()
                    if content and len(content) > 3:  # Minimum content length
                        outline_points.append(content)
                    break
        
        # Fallback: if no patterns matched, try simple line splitting
        if not outline_points:
            potential_points = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
            outline_points = potential_points[:self.max_skeleton_points]
        
        return outline_points

    async def _expand_point_async(self, index: int, point: str, all_points: List[str], 
                                original_question: str, provider_manager, model: str, **kwargs) -> str:
        """Expand a single point asynchronously"""
        context_outline = '\n'.join([f"{i+1}. {p}" for i, p in enumerate(all_points)])
        
        expand_prompt = f"""You are writing one section of a comprehensive answer to the question: "{original_question}"

FULL OUTLINE (for context):
{context_outline}

YOUR SPECIFIC TASK:
Expand ONLY on point {index + 1}: "{point}"

REQUIREMENTS:
- Write a detailed, well-structured section
- Stay focused on this specific point
- Provide concrete details, examples, or explanations
- Write as if this is part of a larger cohesive answer
- Aim for 2-4 paragraphs

DETAILED EXPANSION:"""
        
        response = await provider_manager.generate_response_async(expand_prompt, model, **kwargs)
        return response["content"].strip()

    def _expand_point_sync(self, index: int, point: str, all_points: List[str], 
                          original_question: str, provider_manager, model: str, **kwargs) -> str:
        """Expand a single point synchronously"""
        context_outline = '\n'.join([f"{i+1}. {p}" for i, p in enumerate(all_points)])
        
        expand_prompt = f"""You are writing one section of a comprehensive answer to the question: "{original_question}"

FULL OUTLINE (for context):
{context_outline}

YOUR SPECIFIC TASK:
Expand ONLY on point {index + 1}: "{point}"

REQUIREMENTS:
- Write a detailed, well-structured section
- Stay focused on this specific point
- Provide concrete details, examples, or explanations
- Write as if this is part of a larger cohesive answer
- Aim for 2-4 paragraphs

DETAILED EXPANSION:"""
        
        response = provider_manager.generate_response(expand_prompt, model, **kwargs)
        return response["content"].strip()

    async def _aggregate_parts_async(self, original_question: str, skeleton: str, 
                                   expanded_parts: List[str], provider_manager, 
                                   model: str, **kwargs) -> str:
        """Aggregate expanded parts into coherent answer asynchronously"""
        sections_text = '\n\n--- SECTION BREAK ---\n\n'.join([
            f"SECTION {i+1}:\n{part}" for i, part in enumerate(expanded_parts)
        ])
        
        aggregation_prompt = f"""**ROLE:** You are a final answer synthesizer.
**TASK:** Combine the provided sections into a single, coherent answer and then format it EXACTLY into the specified XML-like structure.

**ORIGINAL QUESTION:** {original_question}
**OUTLINE USED:**
{skeleton}

**SECTIONS TO COMBINE:**
{sections_text}

**INSTRUCTIONS:**
1. First, synthesize all sections into a final, coherent reasoning chain.
2. Place this entire reasoning chain inside the <reasoning> tag.
3. Then, on a new line, state ONLY the final answer to the original question inside the <final> tag.

**REQUIRED FORMAT:**
<reasoning>[Your full, synthesized reasoning here]</reasoning>
<final>[The final answer only]</final>

**FORMATTED OUTPUT:**"""
        
        response = await provider_manager.generate_response_async(aggregation_prompt, model, **kwargs)
        return response["content"].strip()

    def _aggregate_parts_sync(self, original_question: str, skeleton: str, 
                            expanded_parts: List[str], provider_manager, 
                            model: str, **kwargs) -> str:
        """Aggregate expanded parts into coherent answer synchronously"""
        sections_text = '\n\n--- SECTION BREAK ---\n\n'.join([
            f"SECTION {i+1}:\n{part}" for i, part in enumerate(expanded_parts)
        ])
        
        aggregation_prompt = f"""**ROLE:** You are a final answer synthesizer.
**TASK:** Combine the provided sections into a single, coherent answer and then format it EXACTLY into the specified XML-like structure.

**ORIGINAL QUESTION:** {original_question}
**OUTLINE USED:**
{skeleton}

**SECTIONS TO COMBINE:**
{sections_text}

**INSTRUCTIONS:**
1. First, synthesize all sections into a final, coherent reasoning chain.
2. Place this entire reasoning chain inside the <reasoning> tag.
3. Then, on a new line, state ONLY the final answer to the original question inside the <final> tag.

**REQUIRED FORMAT:**
<reasoning>[Your full, synthesized reasoning here]</reasoning>
<final>[The final answer only]</final>

**FORMATTED OUTPUT:**"""
        
        response = provider_manager.generate_response(aggregation_prompt, model, **kwargs)
        return response["content"].strip()

    def _simple_join(self, expanded_parts: List[str]) -> str:
        """Simple joining of parts with double line breaks"""
        return '\n\n'.join(part for part in expanded_parts if part.strip())

    def _fallback_reasoning(self, input_text: str, provider_manager, model: str, 
                          error: str = "Unknown error", **kwargs) -> ReasoningResult:
        """Synchronous fallback to basic Chain-of-Thought reasoning"""
        start_time = time.time()
        
        cot_prompt = f"""Let's think step by step to thoroughly answer the following question:

{input_text}

Please provide a comprehensive, well-structured answer:"""
        
        try:
            response = provider_manager.generate_response(cot_prompt, model, **kwargs)
            final_answer = response["content"]
        except Exception as fallback_error:
            final_answer = f"I apologize, but I encountered errors in both the Skeleton-of-Thought approach and the fallback method. Please try rephrasing your question."
            error = f"SoT Error: {error}; Fallback Error: {str(fallback_error)}"
        
        execution_time = time.time() - start_time
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_trace=f"SoT Fallback Used - Original Error: {error}",
            execution_time=execution_time,
            approach_name=f"{self.name} (Fallback)",
            metadata={
                "fallback_used": True,
                "original_error": error,
                "skeleton_parsing_failed": "parsing" in error.lower()
            }
        )

    async def _fallback_reasoning_async(self, input_text: str, provider_manager, model: str, 
                                      error: str = "Unknown error", **kwargs) -> ReasoningResult:
        """Asynchronous fallback to basic Chain-of-Thought reasoning"""
        start_time = time.time()
        
        cot_prompt = f"""Let's think step by step to thoroughly answer the following question:

{input_text}

Please provide a comprehensive, well-structured answer:"""
        
        try:
            response = await provider_manager.generate_response_async(cot_prompt, model, **kwargs)
            final_answer = response["content"]
        except Exception as fallback_error:
            final_answer = f"I apologize, but I encountered errors in both the Skeleton-of-Thought approach and the fallback method. Please try rephrasing your question."
            error = f"SoT Error: {error}; Fallback Error: {str(fallback_error)}"
        
        execution_time = time.time() - start_time
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_trace=f"SoT Async Fallback Used - Original Error: {error}",
            execution_time=execution_time,
            approach_name=f"{self.name} (Async Fallback)",
            metadata={
                "fallback_used": True,
                "original_error": error,
                "skeleton_parsing_failed": "parsing" in error.lower()
            }
        )

class TreeOfThoughtApproach(BaseReasoningApproach):
    """
    Tree-of-Thought (ToT) - Corrected Implementation
    Source: Yao et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (2023)
    https://arxiv.org/abs/2305.10601
    """
    
    def __init__(self, search_algorithm: str = "bfs", max_depth: int = 3, branching_factor: int = 3, pruning_threshold: float = 7.0):
        super().__init__("Tree-of-Thought (ToT)")
        self.search_algorithm = search_algorithm
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.pruning_threshold = pruning_threshold  # Prune thoughts below this score
    
    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        start_time = time.time()
        
        # Initialize the tree
        root = ThoughtNode(content=f"Initial problem: {input_text}", problem=input_text, depth=0)
        tree = ThoughtTree(root)
        
        # Generate and explore the tree using search algorithms
        if self.search_algorithm == "dfs":
            final_answer = self._dfs_search(tree, provider_manager, model, **kwargs)
        elif self.search_algorithm == "bfs":
            final_answer = self._bfs_search(tree, provider_manager, model, **kwargs)
        else:
            raise ValueError(f"Unknown search algorithm: {self.search_algorithm}")
        
        execution_time = time.time() - start_time
        
        # Build reasoning trace from tree traversal
        reasoning_trace = self._build_trace(tree)
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_trace=reasoning_trace,
            execution_time=execution_time,
            approach_name=self.name,
            metadata={
                "search_algorithm": self.search_algorithm,
                "max_depth": self.max_depth,
                "branching_factor": self.branching_factor,
                "pruning_threshold": self.pruning_threshold,
                "tree_structure": tree.to_dict(),
                "citation": "Yao et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. arXiv:2305.10601"
            }
        )
    
    def _dfs_search(self, tree: 'ThoughtTree', provider_manager, model: str, **kwargs) -> str:
        """Depth-First Search with pruning as per ToT paper"""
        def dfs_explore(node):
            # Base case: reached max depth
            if node.depth >= self.max_depth:
                return
            
            # Generate thoughts (branches) from current node
            thoughts = self._generate_thoughts(node, provider_manager, model, **kwargs)
            
            # Evaluate each thought
            for thought in thoughts:
                thought.evaluation = self._evaluate_thought(thought, provider_manager, model, **kwargs)
                
                # Only add promising thoughts (pruning)
                if thought.evaluation >= self.pruning_threshold:
                    node.add_child(thought)
            
            # Continue DFS on promising children
            # Sort children by evaluation (best first for DFS)
            children = sorted(node.children, key=lambda x: x.evaluation, reverse=True)
            for child in children:
                dfs_explore(child)
        
        # Start DFS exploration
        dfs_explore(tree.root)
        
        # Find best path and generate final answer
        best_path = self._find_best_path(tree.root)
        return self._generate_final_answer(best_path, provider_manager, model, **kwargs)
    
    def _bfs_search(self, tree: 'ThoughtTree', provider_manager, model: str, **kwargs) -> str:
        """Breadth-First Search with pruning as per ToT paper"""
        from collections import deque
        queue = deque([tree.root])
        
        while queue:
            current = queue.popleft()
            
            # Skip if at max depth
            if current.depth >= self.max_depth:
                continue
            
            # Generate thoughts (branches)
            thoughts = self._generate_thoughts(current, provider_manager, model, **kwargs)
            
            # Evaluate and filter thoughts
            for thought in thoughts:
                thought.evaluation = self._evaluate_thought(thought, provider_manager, model, **kwargs)
                
                # Only add promising thoughts (pruning)
                if thought.evaluation >= self.pruning_threshold:
                    current.add_child(thought)
                    queue.append(thought)
        
        # Find best path and generate final answer
        best_path = self._find_best_path(tree.root)
        return self._generate_final_answer(best_path, provider_manager, model, **kwargs)
    
    def _generate_thoughts(self, node: 'ThoughtNode', provider_manager, model: str, **kwargs) -> List['ThoughtNode']:
        """Generate multiple structured thoughts from current node"""
        # Get the reasoning path to this node
        path_to_node = []
        curr = node
        while curr is not None:
            path_to_node.append(curr.content)
            curr = curr.parent
        path_to_node.reverse()
        
        reasoning_path = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(path_to_node))
        
        prompt = f"""Given the problem and current reasoning path, generate {self.branching_factor} different next steps to continue solving this problem.

Problem: {node.problem}

Current reasoning path:
{reasoning_path}

Generate exactly {self.branching_factor} different next steps, each on a separate line starting with a number:
1. [next step option 1]
2. [next step option 2]
3. [next step option 3]

Each step should be a concrete, actionable reasoning step that builds on the current path."""
        
        response = provider_manager.generate_response(prompt, model, **kwargs)
        
        # Parse response into individual thoughts
        thoughts = []
        lines = response["content"].strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, self.branching_factor + 1)):
                # Remove the number prefix
                thought_content = line.split('.', 1)[1].strip() if '.' in line else line
                if thought_content:
                    thought = ThoughtNode(
                        content=thought_content,
                        problem=node.problem,
                        depth=node.depth + 1,
                        parent=node
                    )
                    thoughts.append(thought)
        
        # Ensure we have the right number of thoughts
        return thoughts[:self.branching_factor]
    
    def _evaluate_thought(self, thought: 'ThoughtNode', provider_manager, model: str, **kwargs) -> float:
        """Evaluate a thought using LLM with structured JSON output"""
        # Build complete path to this thought
        path_to_node = []
        curr = thought
        while curr is not None:
            path_to_node.append(curr.content)
            curr = curr.parent
        path_to_node.reverse()
        
        reasoning_path_str = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(path_to_node))
        
        prompt = f"""You are a strict, meticulous fact-checker. Evaluate the LAST step in this reasoning path for its logical and factual correctness.
Return ONLY a JSON object with the score: {{"score": number}}.

SCORING RUBRIC:
- 1-3: The step contains a mathematical error, a logical fallacy, or is a clear dead-end.
- 4-6: The step is plausible but not well-justified or could be a distraction.
- 7-8: The step is logical, correct, and a good continuation of the path.
- 9-10: The step is a critical, highly insightful, and correct step that significantly moves towards the solution.

Critically check all calculations. If there is a calculation error, the score MUST be 3 or lower.

Problem: {thought.problem}

Reasoning Path:
{reasoning_path_str}

Return ONLY a JSON object with the score: {{"score": number}}"""
        
        response = provider_manager.generate_response(prompt, model, **kwargs)
        
        try:
            # Clean response and parse JSON
            content = response["content"].strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split('\n', 1)[1] if '\n' in content else content[3:]
            if content.endswith("```"):
                content = content.rsplit('\n', 1)[0] if '\n' in content else content[:-3]
            
            score_json = json.loads(content.strip())
            score = float(score_json.get("score", 5.0))
            return max(1.0, min(10.0, score))  # Clamp between 1-10
        except Exception as e:
            print(f"Warning: Could not parse evaluation score, using default: {e}")
            return 5.0
    
    def _find_best_path(self, root: 'ThoughtNode') -> List['ThoughtNode']:
        """Find the path with highest cumulative score"""
        if not root:
            return []
        
        best_path = [root]
        best_score = root.evaluation
        
        def dfs_best_path(node, current_path, current_score):
            nonlocal best_path, best_score
            
            # Update best if this is better
            if current_score > best_score:
                best_score = current_score
                best_path = current_path.copy()
            
            # Explore children
            for child in node.children:
                new_path = current_path + [child]
                new_score = current_score + child.evaluation
                dfs_best_path(child, new_path, new_score)
        
        # Start with root's children
        for child in root.children:
            dfs_best_path(child, [root, child], root.evaluation + child.evaluation)
        
        return best_path if best_path else [root]
    
    def _generate_final_answer(self, path: List['ThoughtNode'], provider_manager, model: str, **kwargs) -> str:
        """Generate final answer from best reasoning path"""
        if not path:
            return "Unable to find a valid reasoning path."
        
        reasoning_steps = [node.content for node in path]
        problem = path[0].problem if path else "Unknown problem"
        
        prompt = f"""Based on this step-by-step reasoning path, provide a clear and complete final answer to the problem.

Problem: {problem}

Reasoning Path:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(reasoning_steps))}

Synthesize the reasoning above into a final answer. Be specific and complete:"""
        
        response = provider_manager.generate_response(prompt, model, **kwargs)
        return response["content"].strip()
    
    def _build_trace(self, tree: 'ThoughtTree') -> str:
        """Build comprehensive reasoning trace from tree structure"""
        trace = f"Tree of Thoughts (Search: {self.search_algorithm.upper()})\n"
        trace += f"Problem: {tree.root.problem}\n"
        trace += f"Pruning Threshold: {self.pruning_threshold}\n\n"
        
        # Find best path for highlighting
        best_path = self._find_best_path(tree.root)
        best_path_nodes = set(id(node) for node in best_path)
        
        def build_node_trace(node, indent=0):
            nonlocal trace
            # Mark if this node is on the best path
            marker = "★ " if id(node) in best_path_nodes else "  "
            trace += "  " * indent + f"{marker}Depth {node.depth}: {node.content} (Score: {node.evaluation:.1f})\n"
            
            # Sort children by score for better readability
            children = sorted(node.children, key=lambda x: x.evaluation, reverse=True)
            for child in children:
                build_node_trace(child, indent + 1)
        
        build_node_trace(tree.root)
        
        # Add summary
        trace += f"\n★ Best Path Summary (Total Score: {sum(node.evaluation for node in best_path):.1f}):\n"
        for i, node in enumerate(best_path):
            trace += f"  {i+1}. {node.content} ({node.evaluation:.1f})\n"
        
        return trace


# Node and Tree classes remain the same as in your original implementation
class ThoughtNode:
    """Node in the Tree of Thoughts"""
    
    def __init__(self, content: str, problem: str = "", depth: int = 0, parent=None):
        self.content = content
        self.problem = problem
        self.depth = depth
        self.parent = parent
        self.children = []
        self.evaluation = 0.0
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)


class ThoughtTree:
    """Tree structure for ToT"""
    
    def __init__(self, root: ThoughtNode):
        self.root = root
    
    def to_dict(self):
        """Convert tree to dictionary for serialization"""
        def node_to_dict(node):
            return {
                "content": node.content,
                "depth": node.depth,
                "evaluation": node.evaluation,
                "children": [node_to_dict(child) for child in node.children]
            }
        return node_to_dict(self.root)

class GraphNode:
    """A node in the thought graph representing a partial thought or synthesized artifact."""

    def __init__(self, node_id: int, content: str, node_type: str = "thought", score: Optional[float] = None):
        self.id = node_id
        self.content = content.strip()
        self.type = node_type  # e.g., "problem", "thought", "aggregated", "refined"
        self.score = score
        self.in_edges: List[int] = []
        self.out_edges: List[int] = []
        self.metadata: Dict = {}

    def __repr__(self):
        return f"GraphNode(id={self.id}, type={self.type}, score={self.score}, content={self.content[:60]!r})"


class GraphEdge:
    """Semantic edge between two thoughts."""

    def __init__(self, from_node: int, to_node: int, relation: str = "generates"):
        self.from_node = from_node
        self.to_node = to_node
        self.relation = relation

    def to_dict(self):
        return {"from": self.from_node, "to": self.to_node, "relation": self.relation}


class ThoughtGraph:
    """Container for nodes and edges with helpful utilities."""

    def __init__(self):
        self.nodes: Dict[int, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.next_id = 0

    def add_node(self, content: str, node_type: str = "thought", score: Optional[float] = None) -> GraphNode:
        node = GraphNode(self.next_id, content, node_type, score)
        self.nodes[self.next_id] = node
        self.next_id += 1
        return node

    def add_edge(self, from_id: int, to_id: int, relation: str = "generates"):
        if from_id not in self.nodes or to_id not in self.nodes:
            raise KeyError("Both nodes must exist before adding an edge")
        edge = GraphEdge(from_id, to_id, relation)
        self.edges.append(edge)
        self.nodes[from_id].out_edges.append(to_id)
        self.nodes[to_id].in_edges.append(from_id)

    def get_nodes_by_type(self, node_type: str) -> List[GraphNode]:
        return [n for n in self.nodes.values() if n.type == node_type]

    def to_dict(self):
        return {
            "nodes": {nid: {"content": n.content, "type": n.type, "score": n.score, "metadata": n.metadata} 
                     for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
        }


# ----------------- Utility functions -----------------

def jaccard_similarity(a: str, b: str) -> float:
    """Simple Jaccard similarity over word sets to detect near-duplicate thoughts."""
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    inter = wa.intersection(wb)
    union = wa.union(wb)
    return len(inter) / len(union)


# ----------------- Graph-of-Thoughts Approach -----------------

class GraphOfThoughtsEngine:
    """Graph-of-Thoughts approach with proper sequential building."""

    def __init__(self,
                 beam_width: int = 2,
                 max_nodes: int = 15,
                 reuse_threshold: float = 0.8,
                 score_threshold: float = 6.0,
                 max_iterations: int = 5):
        self.beam_width = beam_width
        self.max_nodes = max_nodes
        self.reuse_threshold = reuse_threshold
        self.score_threshold = score_threshold
        self.max_iterations = max_iterations

    def reason(self, problem: str, provider_manager, model: str, **kwargs) -> Dict:
        """Main entry point. Returns a dict with final answer, graph, and metadata."""
        start_time = time.time()
        graph = ThoughtGraph()
        root = graph.add_node(problem, node_type="problem", score=10.0)

        # Start with problem node in frontier
        frontier: List[int] = [root.id]
        iteration = 0

        print(f"Starting GoT reasoning with max_iterations={self.max_iterations}")

        while iteration < self.max_iterations and len(graph.nodes) < self.max_nodes:
            iteration += 1
            print(f"\n=== ITERATION {iteration} ===")
            print(f"Current frontier: {frontier}")
            print(f"Total nodes: {len(graph.nodes)}")
            
            if not frontier:
                print("Empty frontier - terminating")
                break

            new_candidates: List[Tuple[GraphNode, int, str]] = []

            # Expand each node in the current frontier
            for node_id in frontier:
                if len(graph.nodes) >= self.max_nodes:
                    break
                    
                node = graph.nodes[node_id]
                print(f"\nExpanding Node {node_id} ({node.type}): '{node.content[:100]}...'")

                try:
                    generated_texts = self._generate_for_node(node, problem, provider_manager, model, **kwargs)
                    print(f"Generated {len(generated_texts)} candidate thoughts")
                    
                    for i, txt in enumerate(generated_texts):
                        print(f"  Candidate {i+1}: '{txt[:80]}...'")
                        
                except Exception as e:
                    print(f"ERROR: Generation failed for node {node_id}: {e}")
                    continue

                # Process each generated text
                for txt in generated_texts:
                    txt = self._clean_text(txt)
                    if not txt or len(txt) < 10:
                        continue

                    # Check for reuse
                    reused_id = self._find_reuse_candidate(txt, graph)
                    if reused_id is not None:
                        print(f"  Reusing existing node {reused_id}")
                        try:
                            graph.add_edge(node_id, reused_id, relation="builds_on")
                        except KeyError:
                            pass
                        continue

                    # Create new node
                    candidate = graph.add_node(txt, node_type="thought")
                    try:
                        graph.add_edge(node_id, candidate.id, relation="generates")
                        new_candidates.append((candidate, node_id, "generates"))
                        print(f"  Created Node {candidate.id}: '{txt[:60]}...'")
                    except KeyError:
                        pass

                    if len(graph.nodes) >= self.max_nodes:
                        break

            # Score the new candidates
            if new_candidates:
                print(f"\nScoring {len(new_candidates)} new nodes...")
                nodes_to_score = [c[0] for c in new_candidates]
                try:
                    scores = self._score_nodes(nodes_to_score, problem, provider_manager, model, **kwargs)
                    for node in nodes_to_score:
                        node.score = scores.get(node.id, 5.0)
                        print(f"  Node {node.id} scored: {node.score}")
                except Exception as e:
                    print(f"ERROR: Scoring failed: {e}")
                    for node in nodes_to_score:
                        node.score = 5.0

            # Build next frontier - CRITICAL: Exclude problem node after iteration 1
            next_frontier = []
            
            if iteration == 1:
                # First iteration: use all new thought nodes
                new_thought_nodes = [c[0] for c in new_candidates if c[0].type == "thought"]
                new_thought_nodes.sort(key=lambda x: x.score or 0, reverse=True)
                next_frontier = [n.id for n in new_thought_nodes[:self.beam_width]]
                print(f"After iteration 1: Next frontier from new thoughts: {next_frontier}")
                
            else:
                # Later iterations: select best thought nodes (never include problem node)
                all_thought_nodes = [n for n in graph.nodes.values() 
                                   if n.type in ["thought", "aggregated", "refined"] and n.score is not None]
                all_thought_nodes.sort(key=lambda x: x.score, reverse=True)
                
                # Prioritize recent nodes but include some older high-scoring ones
                recent_nodes = [c[0] for c in new_candidates if c[0].score is not None]
                recent_nodes.sort(key=lambda x: x.score, reverse=True)
                
                # Mix recent and historical
                for n in recent_nodes[:max(1, self.beam_width // 2)]:
                    if len(next_frontier) < self.beam_width:
                        next_frontier.append(n.id)
                        
                for n in all_thought_nodes:
                    if n.id not in next_frontier and len(next_frontier) < self.beam_width:
                        next_frontier.append(n.id)
                        
                print(f"After iteration {iteration}: Next frontier: {next_frontier}")

            frontier = next_frontier

            # Aggregation and refinement
            if iteration >= 2 and iteration % 2 == 0:
                print(f"\nPerforming aggregation/refinement at iteration {iteration}")
                try:
                    self._aggregate_and_refine(graph, provider_manager, model, problem=problem, **kwargs)
                except Exception as e:
                    print(f"WARNING: Aggregation failed: {e}")

            # Show current graph structure
            print(f"\nCurrent graph structure:")
            for edge in graph.edges:
                from_node = graph.nodes[edge.from_node]
                to_node = graph.nodes[edge.to_node]
                print(f"  Node {edge.from_node} ({from_node.type}) --{edge.relation}--> Node {edge.to_node} ({to_node.type})")

        # Final synthesis
        print(f"\n=== SYNTHESIS ===")
        try:
            final_answer = self._synthesize_solution(graph, provider_manager, model, problem=problem, **kwargs)
        except Exception as e:
            print(f"ERROR: Synthesis failed: {e}")
            final_answer = "Unable to synthesize solution due to error."

        execution_time = time.time() - start_time
        print(f"GoT reasoning completed in {execution_time:.2f}s after {iteration} iterations")

        return {
            "final_answer": final_answer,
            "graph": graph,
            "graph_dict": graph.to_dict(),
            "execution_time": execution_time,
            "iterations": iteration,
        }

    def _generate_for_node(self, node: GraphNode, problem: str, provider_manager, model: str, **kwargs) -> List[str]:
        """Generate candidate next steps for a node."""
        if node.type == "problem":
            # Initial breakdown of the problem
            prompt = (
                f"Break down this math problem into 2-3 concrete sequential steps to solve it.\n\n"
                f"PROBLEM: {problem}\n\n"
                f"List 2-3 specific steps (one per line):"
            )
        else:
            # Build on existing thought
            prompt = (
                f"Given this reasoning step, what are 1-2 logical next steps to continue solving the problem?\n\n"
                f"ORIGINAL PROBLEM: {problem}\n"
                f"CURRENT STEP: {node.content}\n\n"
                f"Next 1-2 steps:"
            )

        try:
            resp = provider_manager.generate_response(prompt, model, **kwargs)
            content = resp.get("content", "")
            
            # Parse lines
            lines = []
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Remove numbering/bullets
                cleaned = re.sub(r'^\s*[\d\-\*\u2022\)\.]+\s*', '', line)
                if cleaned and len(cleaned) > 10:
                    lines.append(cleaned)
            
            return lines[:3]  # Max 3 steps
            
        except Exception as e:
            print(f"Generation error: {e}")
            return []

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        text = text.strip()
        text = re.sub(r'^\s*[\d\-\*\u2022\)\.]+\s*', '', text)  # Remove bullets
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        return text

    def _find_reuse_candidate(self, text: str, graph: ThoughtGraph) -> Optional[int]:
        """Find existing similar node."""
        best_id = None
        best_score = 0.0
        
        for nid, node in graph.nodes.items():
            if node.type == "problem":
                continue
            sim = jaccard_similarity(text, node.content)
            if sim > best_score:
                best_score = sim
                best_id = nid
                
        return best_id if best_score >= self.reuse_threshold else None

    def _score_nodes(self, nodes: List[GraphNode], problem: str, provider_manager, model: str, **kwargs) -> Dict[int, float]:
        """Score nodes for quality."""
        scores: Dict[int, float] = {}
        
        for node in nodes:
            prompt = (
                f"Rate this reasoning step for mathematical correctness and helpfulness (1-10).\n\n"
                f"PROBLEM: {problem}\n"
                f"REASONING STEP: {node.content}\n\n"
                f"Return only a JSON object: {{\"score\": X}} where X is 1-10.\n"
                f"1-3: incorrect, 4-6: partially correct, 7-8: good, 9-10: excellent"
            )
            
            try:
                resp = provider_manager.generate_response(prompt, model, **kwargs)
                txt = resp.get("content", "").strip()
                score = self._parse_score(txt)
                scores[node.id] = score
            except Exception:
                scores[node.id] = 5.0
                
        return scores

    def _parse_score(self, text: str) -> float:
        """Extract score from response."""
        try:
            # Try JSON first
            match = re.search(r'\{.*?"score"\s*:\s*([0-9.]+).*?\}', text, re.DOTALL)
            if match:
                return max(1.0, min(10.0, float(match.group(1))))
                
            # Try any number
            match = re.search(r'(\d+(?:\.\d+)?)', text)
            if match:
                return max(1.0, min(10.0, float(match.group(1))))
                
        except ValueError:
            pass
        return 5.0

    def _aggregate_and_refine(self, graph: ThoughtGraph, provider_manager, model: str, problem: str, **kwargs):
        """Perform aggregation and refinement operations."""
        # Find candidates for aggregation
        candidates = [n for n in graph.nodes.values() 
                     if n.type == "thought" and (n.score or 0) >= 6.0]
        candidates.sort(key=lambda x: x.score or 0, reverse=True)
        
        if len(candidates) >= 2:
            a, b = candidates[0], candidates[1]
            if jaccard_similarity(a.content, b.content) < 0.7:  # Not too similar
                prompt = (
                    f"Combine these two reasoning steps into one cohesive step:\n\n"
                    f"Step 1: {a.content}\n"
                    f"Step 2: {b.content}\n\n"
                    f"Combined step:"
                )
                try:
                    resp = provider_manager.generate_response(prompt, model, **kwargs)
                    combined = self._clean_text(resp.get("content", ""))
                    if combined and len(combined) > 15:
                        avg_score = (a.score + b.score) / 2 if a.score and b.score else 6.0
                        agg_node = graph.add_node(combined, node_type="aggregated", score=avg_score)
                        graph.add_edge(a.id, agg_node.id, "aggregated_into")
                        graph.add_edge(b.id, agg_node.id, "aggregated_into")
                        print(f"Aggregated nodes {a.id} and {b.id} into node {agg_node.id}")
                except Exception as e:
                    print(f"Aggregation failed: {e}")

    def _synthesize_solution(self, graph: ThoughtGraph, provider_manager, model: str, problem: str, **kwargs) -> str:
        """Create final answer from graph."""
        # Get high-quality nodes
        quality_nodes = [n for n in graph.nodes.values() 
                        if n.type != "problem" and (n.score or 0) >= 5.0]
        
        if not quality_nodes:
            return "Could not generate quality reasoning steps."
            
        quality_nodes.sort(key=lambda x: x.score or 0, reverse=True)
        
        # Build synthesis prompt
        steps_text = "\n".join([
            f"Step {i+1} (score {n.score:.1f}): {n.content}" 
            for i, n in enumerate(quality_nodes[:6])
        ])
        
        prompt = (
            f"PROBLEM: {problem}\n\n"
            f"REASONING STEPS:\n{steps_text}\n\n"
            f"Using these steps, provide a complete mathematical solution with the final numerical answer. "
            f"Show all calculations clearly and verify the result."
        )
        
        try:
            resp = provider_manager.generate_response(prompt, model, **kwargs)
            return resp.get("content", "").strip()
        except Exception as e:
            return f"Synthesis error: {str(e)}"


class GraphOfThoughtApproach(BaseReasoningApproach):
    """Adapter for GraphOfThoughtsEngine."""
    
    def __init__(self, **engine_kwargs):
        super().__init__("Graph-of-Thought (GoT)")
        self.engine = GraphOfThoughtsEngine(**engine_kwargs)

    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        result_dict = self.engine.reason(input_text, provider_manager, model, **kwargs)
        reasoning_trace = self._build_detailed_trace(result_dict['graph'])

        return ReasoningResult(
            final_answer=result_dict['final_answer'],
            reasoning_trace=reasoning_trace,
            execution_time=result_dict['execution_time'],
            approach_name=self.name,
            metadata={
                "graph_structure": result_dict['graph'].to_dict(),
                "iterations": result_dict.get('iterations', 0),
                "total_nodes": len(result_dict['graph'].nodes),
                "total_edges": len(result_dict['graph'].edges),
                "citation": "Besta et al. (2023). Graph of Thoughts: Solving Elaborate Problems with Large Language Models. arXiv:2308.09687"
            }
        )

    def _build_detailed_trace(self, graph: ThoughtGraph) -> str:
        """Creates a readable string representation of the graph for the trace."""
        if not graph or not graph.nodes:
            return "Graph is empty or was not generated."
        
        trace = f"Graph of Thoughts Reasoning (Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)})\n"
        trace += "=" * 70 + "\n"
        
        # Find problem node
        problem_node = next((n for n in graph.nodes.values() if n.type == "problem"), None)
        if problem_node:
            trace += f"Problem: {problem_node.content}\n\n"
        
        # Sort non-problem nodes by ID
        sorted_nodes = sorted([n for n in graph.nodes.values() if n.type != "problem"], key=lambda n: n.id)

        if not sorted_nodes:
            trace += "No reasoning nodes were generated.\n"
            return trace

        # Group by type
        nodes_by_type = {}
        for node in sorted_nodes:
            if node.type not in nodes_by_type:
                nodes_by_type[node.type] = []
            nodes_by_type[node.type].append(node)

        for node_type, nodes in nodes_by_type.items():
            trace += f"\n{node_type.upper()} NODES:\n" + "-" * 40 + "\n"
            for node in nodes:
                score_str = f" (Score: {node.score:.1f})" if node.score is not None else ""
                trace += f"Node {node.id}{score_str}: {node.content}\n"
                
                # Show connections
                incoming_edges = [e for e in graph.edges if e.to_node == node.id]
                for edge in incoming_edges:
                    source_node = graph.nodes.get(edge.from_node)
                    if source_node:
                        trace += f"  ← {edge.relation} ← Node {edge.from_node} ({source_node.type})\n"
                trace += "\n"
        
        # Statistics
        avg_score = sum(n.score for n in sorted_nodes if n.score is not None) / len([n for n in sorted_nodes if n.score is not None]) if any(n.score for n in sorted_nodes) else 0
        high_quality = len([n for n in sorted_nodes if (n.score or 0) >= 7.0])
        
        trace += f"\nSUMMARY STATISTICS:\n" + "-" * 40 + "\n"
        trace += f"Total reasoning nodes: {len(sorted_nodes)}\n"
        trace += f"Average score: {avg_score:.2f}\n"
        trace += f"High-quality nodes (≥7.0): {high_quality}\n"
        
        return trace

class ReWOOApproach(BaseReasoningApproach):
    """
    ReWOO (Reasoning WithOut Observation)
    Source: Xu et al. "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models" (2023)
    https://arxiv.org/abs/2305.18323
    """
    
    def __init__(self):
        super().__init__("ReWOO")
    
    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        start_time = time.time()
        
        # Step 1: Create reasoning plan without observations
        plan_prompt = f"""Create a reasoning plan for the following problem without making any observations or assumptions:

Problem: {input_text}

Create a plan that focuses on logical reasoning steps:"""
        
        plan_response = provider_manager.generate_response(plan_prompt, model, **kwargs)
        
        # Step 2: Execute reasoning steps
        execute_prompt = f"""Execute the reasoning plan step by step:

Problem: {input_text}

Plan:
{plan_response['content']}

Execute each reasoning step:"""
        
        execute_response = provider_manager.generate_response(execute_prompt, model, **kwargs)
        
        execution_time = time.time() - start_time
        
        reasoning_trace = f"Plan: {plan_response['content']}\n\nExecution: {execute_response['content']}"
        
        return ReasoningResult(
            final_answer=execute_response["content"],
            reasoning_trace=reasoning_trace,
            execution_time=execution_time,
            approach_name=self.name,
            metadata={
                "plan": plan_response,
                "execution": execute_response,
                "citation": "Xu et al. (2023). ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models. arXiv:2305.18323"
            }
        )


class BufferOfThoughtsApproach(BaseReasoningApproach):
    """
    Buffer-of-Thoughts (BoT)
    Source: Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)
    https://arxiv.org/abs/2303.11366
    
    Note: This is a simplified implementation. The original BoT is more complex and involves
    iterative refinement with a buffer of previous attempts.
    """
    
    def __init__(self):
        super().__init__("Buffer-of-Thoughts (BoT)")
    
    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        start_time = time.time()
        
        # Generate multiple thoughts in parallel
        thoughts = []
        for i in range(3):
            thought_prompt = f"""Generate a thought or insight about the following problem:

Problem: {input_text}

Generate thought {i+1}:"""
            
            thought_response = provider_manager.generate_response(thought_prompt, model, **kwargs)
            thoughts.append(thought_response["content"])
        
        # Buffer and synthesize thoughts
        buffer_prompt = f"""Buffer and synthesize these thoughts to solve the problem:

Problem: {input_text}

Thought 1: {thoughts[0]}
Thought 2: {thoughts[1]}
Thought 3: {thoughts[2]}

Synthesize these thoughts and provide the final answer:"""
        
        buffer_response = provider_manager.generate_response(buffer_prompt, model, **kwargs)
        
        execution_time = time.time() - start_time
        
        reasoning_trace = f"Thought 1: {thoughts[0]}\n\nThought 2: {thoughts[1]}\n\nThought 3: {thoughts[2]}\n\nSynthesis: {buffer_response['content']}"
        
        return ReasoningResult(
            final_answer=buffer_response["content"],
            reasoning_trace=reasoning_trace,
            execution_time=execution_time,
            approach_name=self.name,
            metadata={
                "thoughts": thoughts,
                "synthesis": buffer_response,
                "citation": "Shinn et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366"
            }
        )


# Supporting classes for ToT and GoT


class ReasoningApproachManager:
    """Manages different reasoning approaches with proper citations"""
    
    def __init__(self):
        self.approaches = {
            "None": NoneApproach(),
            "Chain-of-Thought (CoT)": ChainOfThoughtApproach(),
            "Least-to-Most Prompting (LtM)": LeastToMostApproach(),
            "Self-Consistency CoT-SC": ChainOfThoughtSCApproach(),
            "Reasoning-as-Planning (RAP)": ReasoningAsPlanningApproach(),
            "Chain-of-Verification (CoVe)": ChainOfVerificationApproach(),
            "Skeleton-of-Thought (SoT)": SkeletonOfThoughtApproach(),
            "Tree-of-Thought (ToT)": TreeOfThoughtApproach(),
            "Graph-of-Thought (GoT)": GraphOfThoughtApproach(),
            "ReWOO": ReWOOApproach(),
            "Buffer-of-Thoughts (BoT)": BufferOfThoughtsApproach()
        }
    
    def get_available_approaches(self) -> List[str]:
        """Get list of available reasoning approaches"""
        return list(self.approaches.keys())
    
    def execute_reasoning(self, approach_name: str, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        """Execute a specific reasoning approach"""
        if approach_name not in self.approaches:
            raise ValueError(f"Unknown reasoning approach: {approach_name}")
        
        return self.approaches[approach_name].reason(input_text, provider_manager, model, **kwargs)
    
    def get_citations(self) -> Dict[str, str]:
        """Get citations for all approaches"""
        citations = {}
        for name, approach in self.approaches.items():
            if hasattr(approach, 'reason'):
                # Get citation from metadata if available
                try:
                    result = approach.reason("test", None, "test")
                    if result.metadata and "citation" in result.metadata:
                        citations[name] = result.metadata["citation"]
                except:
                    citations[name] = "Citation not available"
        return citations
