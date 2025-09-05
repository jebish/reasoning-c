"""
Corrected Reasoning Approaches Implementation
Based on original research papers with proper citations and implementations
"""

import time
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import deque
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

Let's think step by step:"""
        
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
    Least-to-Most Prompting (LtM)
    Source: Zhou et al. "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models" (2022)
    https://arxiv.org/abs/2205.10625
    """
    
    def __init__(self):
        super().__init__("Least-to-Most Prompting (LtM)")
    
    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        start_time = time.time()
        
        # Step 1: Decompose the problem (Reduce step)
        decompose_prompt = f"""Break down the following complex problem into smaller, manageable sub-problems. List them in order from simplest to most complex:

Problem: {input_text}

Sub-problems (from simplest to most complex):"""
        
        decomposition_response = provider_manager.generate_response(decompose_prompt, model, **kwargs)
        
        # Step 2: Solve sub-problems sequentially (Solve step)
        solve_prompt = f"""Now solve each sub-problem step by step, using the results from previous sub-problems:

Original Problem: {input_text}

Sub-problems identified:
{decomposition_response['content']}

Please solve each sub-problem in order and provide the final answer:"""
        
        final_response = provider_manager.generate_response(solve_prompt, model, **kwargs)
        
        execution_time = time.time() - start_time
        
        reasoning_trace = f"Decomposition: {decomposition_response['content']}\n\nSolution: {final_response['content']}"
        
        return ReasoningResult(
            final_answer=final_response["content"],
            reasoning_trace=reasoning_trace,
            execution_time=execution_time,
            approach_name=self.name,
            metadata={
                "decomposition": decomposition_response,
                "final_solution": final_response,
                "citation": "Zhou et al. (2022). Least-to-Most Prompting Enables Complex Reasoning in Large Language Models. arXiv:2205.10625"
            }
        )


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


class SkeletonOfThoughtApproach(BaseReasoningApproach):
    """
    Skeleton-of-Thought (SoT)
    Source: Ning et al. "Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding" (2023)
    https://arxiv.org/abs/2307.15337
    """
    
    def __init__(self):
        super().__init__("Skeleton-of-Thought (SoT)")
    
    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        start_time = time.time()
        
        # Step 1: Create skeleton (Parallel skeleton generation)
        skeleton_prompt = f"""Create a skeleton/outline for answering the following question. Think of the main points and structure without going into details:

Question: {input_text}

Create a skeleton outline with main points:"""
        
        skeleton_response = provider_manager.generate_response(skeleton_prompt, model, **kwargs)
        
        # Step 2: Fill in the skeleton (Parallel point expansion)
        fill_prompt = f"""Now fill in the skeleton with detailed reasoning and provide the final answer:

Question: {input_text}

Skeleton:
{skeleton_response['content']}

Fill in the details and provide the complete answer:"""
        
        fill_response = provider_manager.generate_response(fill_prompt, model, **kwargs)
        
        execution_time = time.time() - start_time
        
        reasoning_trace = f"Skeleton: {skeleton_response['content']}\n\nDetailed Answer: {fill_response['content']}"
        
        return ReasoningResult(
            final_answer=fill_response["content"],
            reasoning_trace=reasoning_trace,
            execution_time=execution_time,
            approach_name=self.name,
            metadata={
                "skeleton": skeleton_response,
                "detailed_answer": fill_response,
                "citation": "Ning et al. (2023). Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding. arXiv:2307.15337"
            }
        )


class TreeOfThoughtApproach(BaseReasoningApproach):
    """
    Tree-of-Thought (ToT)
    Source: Yao et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (2023)
    https://arxiv.org/abs/2305.10601
    
    This implementation includes proper search algorithms (DFS/BFS) as described in the original paper.
    """
    
    def __init__(self, search_algorithm: str = "dfs", max_depth: int = 3, branching_factor: int = 3):
        super().__init__("Tree-of-Thought (ToT)")
        self.search_algorithm = search_algorithm
        self.max_depth = max_depth
        self.branching_factor = branching_factor
    
    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        start_time = time.time()
        
        # Initialize the tree
        root = ThoughtNode(content=input_text, problem=input_text, depth=0)
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
                "tree_structure": tree.to_dict(),
                "citation": "Yao et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. arXiv:2305.10601"
            }
        )
    
    def _dfs_search(self, tree: 'ThoughtTree', provider_manager, model: str, beam_size: int = 3, **kwargs) -> str:
        """Depth-First Search implementation as per ToT paper"""
        stack = [tree.root]
        while stack:
            current = stack.pop()
            if current.depth >= self.max_depth:
                continue
            thoughts = self._generate_thoughts(current, provider_manager, model, **kwargs)
            for thought in thoughts:
                thought.evaluation = self._evaluate_thought(thought, provider_manager, model, **kwargs)
                current.add_child(thought)
            children = sorted(current.children, key=lambda x: x.evaluation, reverse=True)[:beam_size]
            stack.extend(children)
    
    # def _dfs_search(self, tree: 'ThoughtTree', provider_manager, model: str, **kwargs) -> str:
        
    #     stack = [tree.root]
        
    #     while stack:
    #         current = stack.pop()
            
    #         if current.depth >= self.max_depth:
    #             continue
            
    #         # Generate thoughts (branches)
    #         thoughts = self._generate_thoughts(current, provider_manager, model, **kwargs)
            
    #         # Evaluate thoughts
    #         for thought in thoughts:
    #             thought.evaluation = self._evaluate_thought(thought, provider_manager, model, **kwargs)
    #             current.add_child(thought)
            
    #         # Sort by evaluation and add to stack (best first for DFS)
    #         children = sorted(current.children, key=lambda x: x.evaluation, reverse=True)
    #         stack.extend(children)
        
        # Find best path and generate final answer
        best_path = self._find_best_path(tree.root)
        return self._generate_final_answer(best_path, provider_manager, model, **kwargs)
    
    def _bfs_search(self, tree: 'ThoughtTree', provider_manager, model: str, **kwargs) -> str:
        """Breadth-First Search implementation as per ToT paper"""
        queue = deque([tree.root])
        
        while queue:
            current = queue.popleft()
            
            if current.depth >= self.max_depth:
                continue
            
            # Generate thoughts (branches)
            thoughts = self._generate_thoughts(current, provider_manager, model, **kwargs)
            
            # Evaluate thoughts
            for thought in thoughts:
                thought.evaluation = self._evaluate_thought(thought, provider_manager, model, **kwargs)
                current.add_child(thought)
            
            # Add children to queue
            queue.extend(current.children)
        
        # Find best path and generate final answer
        best_path = self._find_best_path(tree.root)
        return self._generate_final_answer(best_path, provider_manager, model, **kwargs)
    
    def _generate_thoughts(self, node: 'ThoughtNode', provider_manager, model: str, **kwargs) -> List['ThoughtNode']:
        """Generate multiple thoughts from current node"""
        prompt = f"""Given the problem and current reasoning, generate {self.branching_factor} different ways to continue thinking about this problem:

Problem: {node.problem}
Current reasoning: {node.content}

Generate {self.branching_factor} different thoughts or approaches:"""
        
        response = provider_manager.generate_response(prompt, model, **kwargs)
        
        # Parse response into individual thoughts
        thoughts = []
        lines = response["content"].split('\n')
        for i, line in enumerate(lines[:self.branching_factor]):
            if line.strip():
                thought = ThoughtNode(
                    content=line.strip(),
                    problem=node.problem,
                    depth=node.depth + 1,
                    parent=node
                )
                thoughts.append(thought)
        
        return thoughts
    
    def _evaluate_thought(self, thought: 'ThoughtNode', provider_manager, model: str, **kwargs) -> float:
        """Evaluate a thought using LLM with structured JSON output"""

        path_to_node = []
        curr = thought # Start from the thought itself
        while curr is not None:
            path_to_node.append(curr.content)
            curr = curr.parent
        path_to_node.reverse()

        reasoning_path_str = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(path_to_node))
        # --- END IMPROVEMENT ---
        
        prompt = f"""Evaluate how promising the LAST step in this reasoning path is for solving the overall problem. 
    Return ONLY a number between 1 and 10 as JSON: {{"score": number}}.
    - A score of 1 means the last step is irrelevant or incorrect.
    - A score of 10 means the last step is a highly logical and promising continuation.

    Problem: {thought.problem}

    Reasoning Path:
    {reasoning_path_str}

    JSON:"""
        response = provider_manager.generate_response(prompt, model, **kwargs)
        try:
            # Be robust to markdown code blocks in the output
            content = response["content"].strip().replace("```json", "").replace("```", "")
            score_json = json.loads(content)
            return float(score_json.get("score", 5))
        except Exception:
            return 5.0
    
    def _find_best_path(self, root: 'ThoughtNode') -> List['ThoughtNode']:
        """Find the best path through the tree"""
        best_path = []
        best_score = 0
        
        def dfs_path(node, current_path, current_score):
            nonlocal best_path, best_score
            
            current_path.append(node)
            current_score += node.evaluation
            
            if not node.children:  # Leaf node
                if current_score > best_score:
                    best_score = current_score
                    best_path = current_path.copy()
            else:
                for child in node.children:
                    dfs_path(child, current_path, current_score)
            
            current_path.pop()
        
        dfs_path(root, [], 0)
        return best_path
    
    def _generate_final_answer(self, path: List['ThoughtNode'], provider_manager, model: str, **kwargs) -> str:
        """Generate final answer from best path"""
        reasoning_steps = [node.content for node in path]
        
        prompt = f"""Based on this reasoning path, provide the final answer:

Problem: {path[0].problem}
Reasoning path:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(reasoning_steps))}

Final answer:"""
        
        response = provider_manager.generate_response(prompt, model, **kwargs)
        return response["content"]
    
    def _build_trace(self, tree: 'ThoughtTree') -> str:
        """Build reasoning trace from tree structure"""
        trace = f"Tree of Thoughts (Search: {self.search_algorithm.upper()})\n"
        trace += f"Problem: {tree.root.problem}\n\n"
        
        def build_node_trace(node, indent=0):
            nonlocal trace
            trace += "  " * indent + f"Depth {node.depth}: {node.content} (Score: {node.evaluation})\n"
            for child in node.children:
                build_node_trace(child, indent + 1)
        
        build_node_trace(tree.root)
        return trace


class GraphOfThoughtApproach(BaseReasoningApproach):
    """
    Graph-of-Thought (GoT)
    Source: Besta et al. "Graph of Thoughts: Solving Elaborate Problems with Large Language Models" (2023)
    https://arxiv.org/abs/2308.09687
    
    This implementation includes proper graph operations as described in the original paper.
    """
    
    def __init__(self, max_nodes: int = 10):
        super().__init__("Graph-of-Thought (GoT)")
        self.max_nodes = max_nodes
    
    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        start_time = time.time()
        
        # Initialize graph
        graph = ThoughtGraph()
        root_node = graph.add_node(input_text, "problem")
        
        # Build the graph through iterative operations
        self._build_graph(graph, root_node, provider_manager, model, **kwargs)
        
        # Traverse graph to find solution
        final_answer = self._traverse_graph(graph, provider_manager, model, **kwargs)
        
        execution_time = time.time() - start_time
        
        # Build reasoning trace
        reasoning_trace = self._build_graph_trace(graph)
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_trace=reasoning_trace,
            execution_time=execution_time,
            approach_name=self.name,
            metadata={
                "max_nodes": self.max_nodes,
                "graph_structure": graph.to_dict(),
                "citation": "Besta et al. (2023). Graph of Thoughts: Solving Elaborate Problems with Large Language Models. arXiv:2308.09687"
            }
        )
    
    def _build_graph(self, graph: 'ThoughtGraph', root_node: 'GraphNode', provider_manager, model: str, **kwargs):
        """Build the graph through GoT operations"""
        current_nodes = [root_node]
        
        for iteration in range(self.max_nodes // 3):  # Limit iterations
            new_nodes = []
            
            for node in current_nodes:
                if len(graph.nodes) >= self.max_nodes:
                    break
                
                # Generate new thoughts
                new_thoughts = self._generate_thoughts(node, provider_manager, model, **kwargs)
                
                for thought_content in new_thoughts:
                    new_node = graph.add_node(thought_content, "thought")
                    graph.add_edge(node.id, new_node.id, "generates")
                    new_nodes.append(new_node)
            
            # Perform graph operations
            self._perform_graph_operations(graph, provider_manager, model, **kwargs)
            current_nodes = new_nodes
    
    def _generate_thoughts(self, node: 'GraphNode', provider_manager, model: str, **kwargs) -> List[str]:
        """Generate new thoughts from a node"""
        prompt = f"""Generate 2-3 new thoughts or insights related to this:

{node.content}

Generate new thoughts:"""
        
        response = provider_manager.generate_response(prompt, model, **kwargs)
        
        # Parse thoughts
        thoughts = []
        lines = response["content"].split('\n')
        for line in lines:
            if line.strip() and not line.strip().startswith(('1.', '2.', '3.', '-', '*')):
                thoughts.append(line.strip())
        
        return thoughts[:3]  # Limit to 3 thoughts
    
    def _perform_graph_operations(self, graph: 'ThoughtGraph', provider_manager, model: str, **kwargs):
        """Perform GoT graph operations: merge, feedback, etc."""
        # Merge operation: combine similar thoughts
        self._merge_operation(graph, provider_manager, model, **kwargs)
        
        # Feedback operation: refine thoughts based on others
        self._feedback_operation(graph, provider_manager, model, **kwargs)
    
    def _merge_operation(self, graph: 'ThoughtGraph', provider_manager, model: str, **kwargs):
        """Merge similar thoughts into a new combined thought"""
        if len(graph.nodes) < 3:
            return
        
        # Find nodes to merge (simplified: take first 2 thought nodes)
        thought_nodes = [n for n in graph.nodes.values() if n.type == "thought"]
        if len(thought_nodes) < 2:
            return
        
        node1, node2 = thought_nodes[0], thought_nodes[1]
        
        prompt = f"""Merge these two thoughts into a single, more comprehensive thought:

Thought 1: {node1.content}
Thought 2: {node2.content}

Merged thought:"""
        
        response = provider_manager.generate_response(prompt, model, **kwargs)
        
        # Create merged node
        merged_node = graph.add_node(response["content"], "merged")
        graph.add_edge(node1.id, merged_node.id, "merged_from")
        graph.add_edge(node2.id, merged_node.id, "merged_from")
    
    def _feedback_operation(self, graph: 'ThoughtGraph', provider_manager, model: str, **kwargs):
        """Create feedback loops to refine thoughts"""
        if len(graph.nodes) < 2:
            return
        
        # Find a thought node to refine
        thought_nodes = [n for n in graph.nodes.values() if n.type == "thought"]
        if not thought_nodes:
            return
        
        node_to_refine = thought_nodes[0]
        
        prompt = f"""Refine this thought based on the overall problem context:

Original thought: {node_to_refine.content}

Refined thought:"""
        
        response = provider_manager.generate_response(prompt, model, **kwargs)
        
        # Create refined node
        refined_node = graph.add_node(response["content"], "refined")
        graph.add_edge(node_to_refine.id, refined_node.id, "refined_to")
    
    def _traverse_graph(self, graph: 'ThoughtGraph', provider_manager, model: str, **kwargs) -> str:
        """Traverse the graph to generate final answer"""
        # Collect all thoughts
        all_thoughts = [node.content for node in graph.nodes.values()]
        
        prompt = f"""Based on all these thoughts and reasoning, provide the final answer:

Problem: {graph.nodes[0].content}

All thoughts and reasoning:
{chr(10).join(f"- {thought}" for thought in all_thoughts[1:])}

Final answer:"""
        
        response = provider_manager.generate_response(prompt, model, **kwargs)
        return response["content"]
    
    def _build_graph_trace(self, graph: 'ThoughtGraph') -> str:
        """Build reasoning trace from graph structure"""
        trace = "Graph of Thoughts\n"
        trace += f"Problem: {graph.nodes[0].content}\n\n"
        
        for node_id, node in graph.nodes.items():
            trace += f"Node {node_id} ({node.type}): {node.content}\n"
            for edge in graph.edges:
                if edge.from_node == node_id:
                    trace += f"  -> {edge.relation} -> Node {edge.to_node}\n"
            trace += "\n"
        
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
    """Node in the Graph of Thoughts"""
    
    def __init__(self, node_id: int, content: str, node_type: str):
        self.id = node_id
        self.content = content
        self.type = node_type  # "problem", "thought", "merged", "refined"


class GraphEdge:
    """Edge in the Graph of Thoughts"""
    
    def __init__(self, from_node: int, to_node: int, relation: str):
        self.from_node = from_node
        self.to_node = to_node
        self.relation = relation


class ThoughtGraph:
    """Graph structure for GoT"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.next_node_id = 0
    
    def add_node(self, content: str, node_type: str) -> GraphNode:
        node = GraphNode(self.next_node_id, content, node_type)
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1
        return node
    
    def add_edge(self, from_node: int, to_node: int, relation: str):
        edge = GraphEdge(from_node, to_node, relation)
        self.edges.append(edge)
    
    def to_dict(self):
        """Convert graph to dictionary for serialization"""
        return {
            "nodes": {str(k): {"content": v.content, "type": v.type} for k, v in self.nodes.items()},
            "edges": [{"from": e.from_node, "to": e.to_node, "relation": e.relation} for e in self.edges]
        }


class ReasoningApproachManager:
    """Manages different reasoning approaches with proper citations"""
    
    def __init__(self):
        self.approaches = {
            "None": NoneApproach(),
            "Chain-of-Thought (CoT)": ChainOfThoughtApproach(),
            "Least-to-Most Prompting (LtM)": LeastToMostApproach(),
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
