import time
import json
from typing import List
from .base import BaseReasoningApproach, ReasoningResult

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