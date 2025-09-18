import time
import re
from typing import Dict, List, Optional, Tuple
from .base import BaseReasoningApproach, ReasoningResult

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