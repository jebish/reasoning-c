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

class GraphOfThoughtApproach(BaseReasoningApproach):
    """
    Enhanced Graph-of-Thought (GoT) implementation
    Based on: Besta et al. "Graph of Thoughts: Solving Elaborate Problems with Large Language Models" (2023)
    """
    
    def __init__(self, max_nodes: int = 15, score_threshold: float = 7.0):
        super().__init__("Enhanced Graph-of-Thought (GoT)")
        self.max_nodes = max_nodes
        self.score_threshold = score_threshold
    
    def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
        start_time = time.time()
        
        # Initialize graph with problem node
        graph = ThoughtGraph()
        root_node = graph.add_node(input_text, "problem", score=8.0)
        
        # Build graph through strategic operations
        self._build_strategic_graph(graph, root_node, provider_manager, model, **kwargs)
        
        # Find best solution path
        final_answer = self._find_best_solution(graph, provider_manager, model, **kwargs)
        
        execution_time = time.time() - start_time
        reasoning_trace = self._build_detailed_trace(graph)
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_trace=reasoning_trace,
            execution_time=execution_time,
            approach_name=self.name,
            metadata={
                "max_nodes": self.max_nodes,
                "score_threshold": self.score_threshold,
                "graph_structure": graph.to_dict(),
                "best_path_scores": self._get_path_scores(graph),
                "citation": "Besta et al. (2023). Graph of Thoughts: Solving Elaborate Problems with Large Language Models. arXiv:2308.09687"
            }
        )
    
    def _build_strategic_graph(self, graph: ThoughtGraph, root_node: GraphNode, provider_manager, model: str, **kwargs):
        """Build graph using strategic GoT operations"""
        active_nodes = [root_node]
        iteration = 0
        
        while len(graph.nodes) < self.max_nodes and active_nodes and iteration < 5:
            iteration += 1
            new_nodes = []
            
            # Generate phase
            for node in active_nodes:
                if len(graph.nodes) >= self.max_nodes:
                    break
                
                generated_thoughts = self._generate_strategic_thoughts(
                    node, graph.nodes[0].content, provider_manager, model, **kwargs
                )
                
                for thought_content in generated_thoughts:
                    new_node = graph.add_node(thought_content, "thought")
                    graph.add_edge(node.id, new_node.id, "generates")
                    new_nodes.append(new_node)
            
            # Score new thoughts
            if new_nodes:
                scores = self._score_thoughts(new_nodes, provider_manager, model, **kwargs)
                for node in new_nodes:
                    node.score = scores.get(node.id, 5.0)
            
            # Select best thoughts for next iteration
            all_unscored = [n for n in graph.get_nodes_by_type("thought") if n.score >= self.score_threshold]
            active_nodes = sorted(all_unscored, key=lambda x: x.score, reverse=True)[:3]
            
            # Perform graph operations
            if iteration % 2 == 0:  # Every other iteration
                self._perform_advanced_operations(graph, provider_manager, model, **kwargs)
    
    def _generate_strategic_thoughts(self, node: GraphNode, problem_context: str, provider_manager, model: str, **kwargs) -> List[str]:
        """Generate thoughts using strategic approaches"""
        strategies = [
            "decompose this into smaller, manageable parts",
            "explore alternative approaches or perspectives", 
            "identify potential challenges or obstacles",
            "consider what resources or information might be needed"
        ]
        
        thoughts = []
        for strategy in strategies[:2]:  # Use 2 strategies per node
            prompt = f"""Problem context: {problem_context}

Current thought: {node.content}

Strategy: {strategy}

Generate a new thought following this strategy:"""
            
            response = provider_manager.generate_response(prompt, model, **kwargs)
            thought = response["content"].strip()
            if thought and len(thought) > 10:  # Basic quality check
                thoughts.append(thought)
        
        return thoughts
    
    def _score_thoughts(self, thoughts: List[GraphNode], provider_manager, model: str, **kwargs) -> Dict[int, float]:
        """Score thoughts based on quality, relevance, and feasibility"""
        scores = {}
        
        for node in thoughts:
            prompt = f"""Rate this thought on a scale of 1-10 considering:
- Relevance to the problem
- Clarity and specificity  
- Feasibility/practicality
- Potential to lead to solution

Thought: {node.content}

Provide only a single number (1-10):"""
            
            response = provider_manager.generate_response(prompt, model, **kwargs)
            score = self._parse_score(response["content"])
            scores[node.id] = score
            
        return scores
    
    def _parse_score(self, response: str) -> float:
        """Extract numeric score from response"""
        import re
        numbers = re.findall(r'\b([1-9](?:\.[0-9])?|10(?:\.0)?)\b', response)
        if numbers:
            return float(numbers[0])
        return 5.0  # Default score
    
    def _perform_advanced_operations(self, graph: ThoughtGraph, provider_manager, model: str, **kwargs):
        """Perform advanced GoT operations: aggregate, refine, validate"""
        
        # Aggregate operation: combine high-scoring thoughts
        high_scoring = [n for n in graph.get_nodes_by_type("thought") if n.score >= self.score_threshold]
        if len(high_scoring) >= 2:
            self._aggregate_thoughts(graph, high_scoring[:2], provider_manager, model, **kwargs)
        
        # Refinement operation
        thoughts_to_refine = [n for n in graph.get_nodes_by_type("thought") if 6.0 <= n.score < self.score_threshold]
        if thoughts_to_refine:
            self._refine_thought(graph, thoughts_to_refine[0], provider_manager, model, **kwargs)
    
    def _aggregate_thoughts(self, graph: ThoughtGraph, thoughts: List[GraphNode], provider_manager, model: str, **kwargs):
        """Combine multiple thoughts into a more comprehensive one"""
        thought_contents = [t.content for t in thoughts]
        
        prompt = f"""Combine these thoughts into a single, more comprehensive and actionable thought:

{chr(10).join(f"- {content}" for content in thought_contents)}

Combined thought:"""
        
        response = provider_manager.generate_response(prompt, model, **kwargs)
        
        # Create aggregated node
        agg_node = graph.add_node(response["content"], "aggregated")
        for thought in thoughts:
            graph.add_edge(thought.id, agg_node.id, "aggregated_into")
        
        # Score the aggregated thought
        scores = self._score_thoughts([agg_node], provider_manager, model, **kwargs)
        agg_node.score = scores.get(agg_node.id, 7.0)
    
    def _refine_thought(self, graph: ThoughtGraph, thought: GraphNode, provider_manager, model: str, **kwargs):
        """Refine a thought to improve its quality"""
        prompt = f"""Improve and refine this thought to make it clearer, more specific, and more actionable:

Original: {thought.content}

Refined version:"""
        
        response = provider_manager.generate_response(prompt, model, **kwargs)
        
        refined_node = graph.add_node(response["content"], "refined")
        graph.add_edge(thought.id, refined_node.id, "refined_to")
        
        # Score refined thought
        scores = self._score_thoughts([refined_node], provider_manager, model, **kwargs)
        refined_node.score = scores.get(refined_node.id, thought.score + 1.0)
    
    def _find_best_solution(self, graph: ThoughtGraph, provider_manager, model: str, **kwargs) -> str:
        """Find the best solution by traversing highest-scoring path"""
        
        # Get all high-quality thoughts
        quality_thoughts = [n for n in graph.nodes.values() 
                          if n.type in ["thought", "aggregated", "refined"] and n.score >= self.score_threshold]
        
        if not quality_thoughts:
            quality_thoughts = sorted([n for n in graph.nodes.values() if n.type != "problem"], 
                                    key=lambda x: getattr(x, 'score', 5.0), reverse=True)[:5]
        
        # Generate solution based on best thoughts
        best_thoughts_content = [t.content for t in sorted(quality_thoughts, key=lambda x: x.score, reverse=True)]
        
        prompt = f"""Based on this high-quality reasoning, provide the final solution:

Original Problem: {graph.nodes[0].content}

Best thoughts and insights:
{chr(10).join(f"• {content}" for content in best_thoughts_content)}

Provide a clear, actionable final solution:"""
        
        response = provider_manager.generate_response(prompt, model, **kwargs)
        return response["content"]
    
    def _get_path_scores(self, graph: ThoughtGraph) -> Dict:
        """Get scoring information for metadata"""
        scores_by_type = {}
        for node in graph.nodes.values():
            node_type = node.type
            if node_type not in scores_by_type:
                scores_by_type[node_type] = []
            scores_by_type[node_type].append(getattr(node, 'score', 0.0))
        
        return {k: {"avg": sum(v)/len(v), "max": max(v), "count": len(v)} 
                for k, v in scores_by_type.items() if v}
    
    def _build_detailed_trace(self, graph: ThoughtGraph) -> str:
        """Build comprehensive reasoning trace"""
        trace = "Enhanced Graph of Thoughts Reasoning\n"
        trace += "=" * 50 + "\n\n"
        trace += f"Problem: {graph.nodes[0].content}\n\n"
        
        # Group nodes by type and score
        by_type = {}
        for node in graph.nodes.values():
            if node.type not in by_type:
                by_type[node.type] = []
            by_type[node.type].append(node)
        
        for node_type, nodes in by_type.items():
            if node_type == "problem":
                continue
                
            trace += f"{node_type.upper()} NODES:\n"
            trace += "-" * 20 + "\n"
            
            # Sort by score if available
            sorted_nodes = sorted(nodes, key=lambda x: getattr(x, 'score', 0), reverse=True)
            
            for node in sorted_nodes:
                score_str = f" (Score: {getattr(node, 'score', 'N/A')})" if hasattr(node, 'score') else ""
                trace += f"Node {node.id}{score_str}: {node.content}\n"
                
                # Show relationships
                related_edges = [e for e in graph.edges if e.from_node == node.id or e.to_node == node.id]
                for edge in related_edges:
                    if edge.from_node == node.id:
                        trace += f"  → {edge.relation} → Node {edge.to_node}\n"
                    else:
                        trace += f"  ← {edge.relation} ← Node {edge.from_node}\n"
                trace += "\n"
            
            trace += "\n"
        
        return trace

class GraphNode:
    """Enhanced node in the Graph of Thoughts with scoring capability"""
    
    def __init__(self, node_id: int, content: str, node_type: str, score: float = None):
        self.id = node_id
        self.content = content
        self.type = node_type  # "problem", "thought", "aggregated", "refined"
        self.score = score  # Quality score for the thought

class ThoughtGraph:
    """Enhanced graph structure for GoT with additional utility methods"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.next_node_id = 0
    
    def add_node(self, content: str, node_type: str, score: float = None) -> GraphNode:
        node = GraphNode(self.next_node_id, content, node_type, score)
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1
        return node
    
    def add_edge(self, from_node: int, to_node: int, relation: str):
        edge = GraphEdge(from_node, to_node, relation)
        self.edges.append(edge)
    
    def get_nodes_by_type(self, node_type: str) -> List[GraphNode]:
        """Get all nodes of a specific type"""
        return [node for node in self.nodes.values() if node.type == node_type]
    
    def get_neighbors(self, node_id: int) -> List[GraphNode]:
        """Get neighboring nodes"""
        neighbors = []
        for edge in self.edges:
            if edge.from_node == node_id:
                neighbors.append(self.nodes[edge.to_node])
            elif edge.to_node == node_id:
                neighbors.append(self.nodes[edge.from_node])
        return neighbors
    
    def to_dict(self):
        """Convert graph to dictionary for serialization"""
        return {
            "nodes": {
                str(k): {
                    "content": v.content, 
                    "type": v.type,
                    "score": getattr(v, 'score', None)
                } for k, v in self.nodes.items()
            },
            "edges": [{"from": e.from_node, "to": e.to_node, "relation": e.relation} for e in self.edges]
        }

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
