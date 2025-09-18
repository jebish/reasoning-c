import re
import time
from typing import Dict, Any
from .base import BaseReasoningApproach, ReasoningResult

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