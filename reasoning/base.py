import re
import json
import time
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

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