import time
import re
import asyncio
from typing import List
from .base import BaseReasoningApproach, ReasoningResult

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
                full_reasoning_trace += f"\n--- Expanding Point {i+1}: --- \n{result}\n"
        
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
                full_reasoning_trace += f"\n--- Expanding Point {i+1}: --- \n{expanded_part}\n"
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