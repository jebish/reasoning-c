"""
Corrected Reasoning Approaches Implementation
Based on original research papers with proper citations and implementations
"""

import time
import json
import random
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict, deque
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

# class GraphNode:
#     """A node in the thought graph representing a partial thought or synthesized artifact."""

#     def __init__(self, node_id: int, content: str, node_type: str = "thought", score: Optional[float] = None):
#         self.id = node_id
#         self.content = content.strip()
#         self.type = node_type  # e.g., "problem", "thought", "aggregated", "refined"
#         self.score = score
#         self.in_edges: List[int] = []
#         self.out_edges: List[int] = []
#         self.metadata: Dict = {}

#     def __repr__(self):
#         return f"GraphNode(id={self.id}, type={self.type}, score={self.score}, content={self.content[:60]!r})"


# class GraphEdge:
#     """Semantic edge between two thoughts."""

#     def __init__(self, from_node: int, to_node: int, relation: str = "generates"):
#         self.from_node = from_node
#         self.to_node = to_node
#         self.relation = relation

#     def to_dict(self):
#         return {"from": self.from_node, "to": self.to_node, "relation": self.relation}


# class ThoughtGraph:
#     """Container for nodes and edges with helpful utilities."""

#     def __init__(self):
#         self.nodes: Dict[int, GraphNode] = {}
#         self.edges: List[GraphEdge] = []
#         self.next_id = 0

#     def add_node(self, content: str, node_type: str = "thought", score: Optional[float] = None) -> GraphNode:
#         node = GraphNode(self.next_id, content, node_type, score)
#         self.nodes[self.next_id] = node
#         self.next_id += 1
#         return node

#     def add_edge(self, from_id: int, to_id: int, relation: str = "generates"):
#         if from_id not in self.nodes or to_id not in self.nodes:
#             raise KeyError("Both nodes must exist before adding an edge")
#         edge = GraphEdge(from_id, to_id, relation)
#         self.edges.append(edge)
#         self.nodes[from_id].out_edges.append(to_id)
#         self.nodes[to_id].in_edges.append(from_id)

#     def get_nodes_by_type(self, node_type: str) -> List[GraphNode]:
#         return [n for n in self.nodes.values() if n.type == node_type]

#     def to_dict(self):
#         return {
#             "nodes": {nid: {"content": n.content, "type": n.type, "score": n.score} for nid, n in self.nodes.items()},
#             "edges": [e.to_dict() for e in self.edges],
#         }


# # ----------------- Utility functions -----------------

# def jaccard_similarity(a: str, b: str) -> float:
#     """Simple Jaccard similarity over word sets to detect near-duplicate thoughts.
#     Lightweight and deterministic; replace with embedding similarity if available.
#     """
#     wa = set(re.findall(r"\w+", a.lower()))
#     wb = set(re.findall(r"\w+", b.lower()))
#     if not wa or not wb:
#         return 0.0
#     inter = wa.intersection(wb)
#     union = wa.union(wb)
#     return len(inter) / len(union)


# # ----------------- Graph-of-Thoughts Approach -----------------

# class GraphOfThoughtsEngine:
#     """Refactored Graph-of-Thoughts approach with flexible policies.

#     Parameters
#     ----------
#     beam_width: int
#         How many promising nodes to keep for expansion each iteration (beam search).
#     max_nodes: int
#         Global cap on graph size to avoid runaway generation.
#     reuse_threshold: float
#         If a newly generated thought has similarity >= reuse_threshold with any existing node,
#         we link to the existing node instead of creating a duplicate.
#     score_threshold: float
#         Minimum score for a node to be considered "high quality" when synthesizing.
#     max_iterations: int
#         Safety cap on number of expansion iterations.
#     """

#     def __init__(self,
#                  beam_width: int = 4,
#                  max_nodes: int = 40,
#                  reuse_threshold: float = 0.7,
#                  score_threshold: float = 7.0,
#                  max_iterations: int = 8):
#         self.beam_width = beam_width
#         self.max_nodes = max_nodes
#         self.reuse_threshold = reuse_threshold
#         self.score_threshold = score_threshold
#         self.max_iterations = max_iterations

#     def reason(self, problem: str, provider_manager, model: str, **kwargs) -> Dict:
#         """Main entry point. Returns a dict with final answer, graph, and metadata."""
#         start_time = time.time()
#         graph = ThoughtGraph()
#         root = graph.add_node(problem, node_type="problem", score=10.0)

#         # Active frontier: list of node ids considered for expansion (beam)
#         frontier: List[int] = [root.id]

#         iteration = 0
#         while iteration < self.max_iterations and len(graph.nodes) < self.max_nodes and frontier:
#             iteration += 1
#             new_candidates: List[Tuple[GraphNode, int, str]] = []  # (node, parent_id, relation)

#             # Generate successors for each node in frontier
#             for node_id in frontier:
#                 if len(graph.nodes) >= self.max_nodes:
#                     break
#                 node = graph.nodes[node_id]

#                 # Use provider to generate a small batch of candidate next thoughts
#                 generated_texts = self._generate_for_node(node, problem, provider_manager, model, **kwargs)

#                 for txt in generated_texts:
#                     txt = self._clean_text(txt)
#                     if not txt:
#                         continue

#                     # Check reuse against existing nodes
#                     reused_id = self._find_reuse_candidate(txt, graph)
#                     if reused_id is not None:
#                         # Add an edge to the existing node instead of creating a duplicate
#                         graph.add_edge(node_id, reused_id, relation="reuses")
#                         continue

#                     # Otherwise create a tentative node (score will be filled by scorer)
#                     candidate = graph.add_node(txt, node_type="thought")
#                     graph.add_edge(node_id, candidate.id, relation="generates")
#                     new_candidates.append((candidate, node_id, "generates"))

#                     if len(graph.nodes) >= self.max_nodes:
#                         break

#             # Score newly created nodes (batch-friendly hook)
#             if new_candidates:
#                 nodes_to_score = [c[0] for c in new_candidates]
#                 scores = self._score_nodes(nodes_to_score, problem, provider_manager, model, **kwargs)
#                 for node in nodes_to_score:
#                     node.score = scores.get(node.id, 5.0)

#             # Build next frontier using beam policy (top-K by score among recent nodes)
#             recent_ids = [c[0].id for c in new_candidates]
#             scored_recent = [graph.nodes[nid] for nid in recent_ids if graph.nodes[nid].score is not None]
#             scored_recent.sort(key=lambda x: x.score, reverse=True)

#             # Keep some previously promising nodes as well (diversity)
#             previous_promising = [n for n in graph.nodes.values() if (n.score or 0) >= self.score_threshold and n.type == "thought"]
#             previous_promising_ids = sorted(previous_promising, key=lambda x: x.score or 0, reverse=True)[:self.beam_width]

#             next_frontier_ids = [n.id for n in scored_recent[: self.beam_width]]
#             # fill with previous promising if frontier is small
#             for pn in previous_promising_ids:
#                 if pn.id not in next_frontier_ids:
#                     next_frontier_ids.append(pn.id)
#                 if len(next_frontier_ids) >= self.beam_width:
#                     break

#             frontier = next_frontier_ids

#             # optionally perform higher-order ops (aggregate/refine) every few iterations
#             if iteration % 2 == 0:
#                 self._aggregate_and_refine(graph, provider_manager, model, problem=problem, **kwargs)

#             # Stopping condition: if we have one or more nodes above a high-confidence threshold
#             high_conf_nodes = [n for n in graph.nodes.values() if (n.score or 0) >= 9.0]
#             if high_conf_nodes:
#                 # We have at least one highly confident thought; stop early to synthesize
#                 break

#         # Synthesize final answer from high-quality subgraph
#         final_answer = self._synthesize_solution(graph, provider_manager, model, problem=problem, **kwargs)
#         execution_time = time.time() - start_time

#         return {
#             "final_answer": final_answer,
#             "graph": graph,
#             "graph_dict": graph.to_dict(),
#             "execution_time": execution_time,
#             "iterations": iteration,
#         }

#     # ----------------- generation, scoring, reuse, and ops -----------------

#     def _generate_for_node(self, node: GraphNode, problem: str, provider_manager, model: str, **kwargs) -> List[str]:
#         """Generate candidate next steps for a node.

#         This method is intentionally small and strict: it asks for 2-4 concrete steps. Providers may be batched.
#         """
#         branch = max(2, min(4, self.beam_width))
#         prompt = (
#             f"You are a concise reasoning engine. Provide exactly {branch} concrete next steps (one per line)"
#             f" that advance the solution to the problem.\n\nPROBLEM: {problem}\nCURRENT THOUGHT: {node.content}\n\nSTEPS:" 
#         )
#         resp = provider_manager.generate_response(prompt, model, **kwargs)
#         content = resp.get("content", "")
#         lines = [ln.strip() for ln in re.split(r"\r?\n", content) if ln.strip()]
#         # extract numbered or bullet lines if present
#         extracted = []
#         for ln in lines:
#             # drop leading markers like 1. - •
#             cleaned = re.sub(r'^\s*[\d\-\*\u2022\)\.]+\s*', '', ln)
#             if cleaned:
#                 extracted.append(cleaned)
#         # If LLM returned one sentence or paragraph, try to split into clauses using semicolons
#         if len(extracted) == 1 and ';' in extracted[0]:
#             parts = [p.strip() for p in extracted[0].split(';') if p.strip()]
#             extracted = parts[:branch]
#         return extracted[:branch]

#     def _clean_text(self, text: str) -> str:
#         """Remove conversational fluff and keep a concise single-step thought."""
#         # Remove common lead-in phrases
#         text = text.strip()
#         # Drop markdown fences if present
#         text = re.sub(r"^```.*?```$", "", text, flags=re.DOTALL).strip()
#         # remove leading enumerations like '1.' or '-'
#         text = re.sub(r'^\s*[\d\-\*\u2022\)\.]+\s*', '', text)
#         # collapse whitespace
#         text = re.sub(r"\s+", " ", text)
#         return text

#     def _find_reuse_candidate(self, text: str, graph: ThoughtGraph) -> Optional[int]:
#         """Return an existing node id if similarity >= reuse_threshold, else None."""
#         best_id = None
#         best_score = 0.0
#         for nid, node in graph.nodes.items():
#             if node.type == "problem":
#                 continue
#             sim = jaccard_similarity(text, node.content)
#             if sim > best_score:
#                 best_score = sim
#                 best_id = nid
#         if best_score >= self.reuse_threshold:
#             return best_id
#         return None

#     def _score_nodes(self, nodes: List[GraphNode], problem: str, provider_manager, model: str, **kwargs) -> Dict[int, float]:
#         """Score nodes using the LLM evaluator. Returns dict node_id->score.

#         This method keeps a simple, strict JSON output protocol and is batch-ready (call provider in loop or batched API).
#         """
#         scores: Dict[int, float] = {}
#         for node in nodes:
#             prompt = (
#                 f"You are a strict fact-checker. Evaluate the following single reasoning step for its logical/factual correctness"
#                 f" in the context of the PROBLEM: {problem}\n\nSTEP: {node.content}\n\n"
#                 "Return ONLY a JSON object like {\"score\": 1-10} and nothing else.\n"
#                 "Scoring: 1-3 incorrect; 4-6 plausible; 7-8 good; 9-10 excellent.\n"
#             )
#             resp = provider_manager.generate_response(prompt, model, **kwargs)
#             txt = resp.get("content", "").strip()
#             score = self._parse_score_json_like(txt)
#             scores[node.id] = float(score)
#         return scores

#     def _parse_score_json_like(self, text: str) -> float:
#         # extract first JSON object
#         try:
#             m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
#             if not m:
#                 # try to extract single number
#                 num = re.search(r"(\d+(?:\.\d+)?)", text)
#                 if num:
#                     return float(num.group(1))
#                 return 5.0
#             obj = json.loads(m.group(0))
#             return float(obj.get("score", 5.0))
#         except Exception:
#             return 5.0

#     def _aggregate_and_refine(self, graph: ThoughtGraph, provider_manager, model: str, problem: str, **kwargs):
#         """Find top candidate nodes and attempt aggregation/refinement.
#         This is deliberately simple: pick top-2 distinct high-scoring nodes for aggregation.
#         """
#         candidates = [n for n in graph.nodes.values() if n.type == "thought" and (n.score or 0) >= (self.score_threshold - 1)]
#         candidates = sorted(candidates, key=lambda x: x.score or 0, reverse=True)
#         if len(candidates) >= 2:
#             a, b = candidates[0], candidates[1]
#             # synthesize
#             prompt = (
#                 f"Combine the following two concise steps into a single denser, precise step.\n\n"
#                 f"Step A: {a.content}\nStep B: {b.content}\n\nReturn only the single combined step."
#             )
#             resp = provider_manager.generate_response(prompt, model, **kwargs)
#             txt = resp.get("content", "").strip()
#             txt = self._clean_text(txt)
#             if txt:
#                 agg = graph.add_node(txt, node_type="aggregated", score=(a.score + b.score) / 2 if a.score and b.score else None)
#                 graph.add_edge(a.id, agg.id, "aggregated_into")
#                 graph.add_edge(b.id, agg.id, "aggregated_into")

#         # refine a marginal node
#         to_refine = [n for n in graph.nodes.values() if n.type == "thought" and 5.5 <= (n.score or 0) < self.score_threshold]
#         if to_refine:
#             candidate = sorted(to_refine, key=lambda x: x.score or 0, reverse=True)[0]
#             prompt = (
#                 f"Rewrite the following step to be more precise and remove ambiguity. Return only the refined step.\n\nOriginal: {candidate.content}"
#             )
#             resp = provider_manager.generate_response(prompt, model, **kwargs)
#             txt = self._clean_text(resp.get("content", ""))
#             if txt:
#                 ref = graph.add_node(txt, node_type="refined", score=(candidate.score or 5.0) + 0.5)
#                 graph.add_edge(candidate.id, ref.id, "refined_to")

#     def _synthesize_solution(self, graph: ThoughtGraph, provider_manager, model: str, problem: str, **kwargs) -> str:
#         """Create final answer from top-scoring nodes and optionally short paths connecting them.

#         Strategy:
#         - select top-K nodes (by score)
#         - optionally extend them by following in_edges to build a small subgraph
#         - ask the model to synthesize a step-by-step solution using those nodes
#         """
#         quality_nodes = [n for n in graph.nodes.values() if n.type != "problem" and (n.score or 0) >= self.score_threshold]
#         if not quality_nodes:
#             # relax threshold
#             quality_nodes = sorted([n for n in graph.nodes.values() if n.type != "problem"], key=lambda x: x.score or 0, reverse=True)[: min(6, len(graph.nodes))]

#         if not quality_nodes:
#             return "Could not generate a high-quality solution."

#         # Build compact context lines
#         header = f"Problem: {graph.nodes[0].content}\n\nUse the following high-quality reasoning steps (do NOT add new assumptions)."
#         steps = []
#         for n in sorted(quality_nodes, key=lambda x: x.score or 0, reverse=True):
#             score_str = f"(score={n.score:.1f})" if n.score is not None else "(score=N/A)"
#             steps.append(f"{score_str} {n.content}")

#         prompt = header + "\n\nHigh-quality steps:\n" + "\n".join(f"- {s}" for s in steps) + (
#             "\n\nInstruction: Synthesize the above into a clear, step-by-step solution and compute the final answer. Be concise and do not invent extra facts."
#         )
#         resp = provider_manager.generate_response(prompt, model, **kwargs)
#         return resp.get("content", "").strip()

# class GraphOfThoughtApproach(BaseReasoningApproach):
#     """
#     Adapter for the GraphOfThoughtsEngine to make it compatible with the BaseReasoningApproach framework.
#     """
#     def __init__(self, **engine_kwargs):
#         super().__init__("Graph-of-Thought (GoT)")
#         # This class now holds an instance of the powerful engine
#         self.engine = GraphOfThoughtsEngine(**engine_kwargs)

#     def reason(self, input_text: str, provider_manager, model: str, **kwargs) -> ReasoningResult:
#         # Call the engine's reason method, which returns a dictionary
#         result_dict = self.engine.reason(input_text, provider_manager, model, **kwargs)

#         # Build the detailed trace from the graph object
#         reasoning_trace = self._build_detailed_trace(result_dict['graph'])

#         # Package the dictionary's results into the expected ReasoningResult object
#         return ReasoningResult(
#             final_answer=result_dict['final_answer'],
#             reasoning_trace=reasoning_trace,
#             execution_time=result_dict['execution_time'],
#             approach_name=self.name,
#             metadata={
#                 "graph_structure": result_dict['graph'].to_dict(),
#                 "iterations": result_dict.get('iterations', 0),
#                 "citation": "Besta et al. (2023). Graph of Thoughts: Solving Elaborate Problems with Large Language Models. arXiv:2308.09687"
#             }
#         )

#     def _build_detailed_trace(self, graph: ThoughtGraph) -> str:
#         """Creates a readable string representation of the graph for the trace."""
#         if not graph or not graph.nodes:
#             return "Graph is empty or was not generated."
        
#         trace = f"Graph of Thoughts Reasoning (Nodes: {len(graph.nodes)})\n" + "=" * 50 + "\n"
#         trace += f"Problem: {graph.nodes[0].content}\n\n"
        
#         # Sort nodes by ID for a chronological view
#         sorted_nodes = sorted(graph.nodes.values(), key=lambda n: n.id)

#         for node in sorted_nodes:
#             if node.type == "problem": continue
#             score_str = f" (Score: {node.score:.1f})" if node.score is not None else ""
#             trace += f"Node {node.id} [{node.type.upper()}]{score_str}: {node.content}\n"
#             # Show incoming connections that formed this node
#             for edge in graph.edges:
#                 if edge.to_node == node.id:
#                     trace += f"  ← {edge.relation} ← Node {edge.from_node} ({graph.nodes[edge.from_node].type})\n"
#             trace += "\n"
#         return trace

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
