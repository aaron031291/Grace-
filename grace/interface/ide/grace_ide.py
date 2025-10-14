"""
Grace IDE - Live Development Environment
Block-card representation with visual flow editor and sandbox execution.
"""

import uuid
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BlockType(Enum):
    """Types of blocks available in the IDE."""

    DATA_SOURCE = "data_source"
    TRANSFORMATION = "transformation"
    ANALYSIS = "analysis"
    API_CALL = "api_call"
    CONDITION = "condition"
    OUTPUT = "output"


@dataclass
class BlockInput:
    """Input parameter for a block."""

    name: str
    data_type: str
    required: bool = True
    default_value: Any = None
    description: str = ""


@dataclass
class BlockOutput:
    """Output parameter from a block."""

    name: str
    data_type: str
    description: str = ""


@dataclass
class FlowBlock:
    """Individual block in the visual flow."""

    block_id: str
    name: str
    description: str
    block_type: BlockType
    inputs: List[BlockInput]
    outputs: List[BlockOutput]
    position: Dict[str, float]  # {"x": 100, "y": 200}
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualFlow:
    """Complete visual flow/pipeline."""

    flow_id: str
    name: str
    description: str
    blocks: List[FlowBlock]
    connections: List[Dict[str, str]]
    creator_id: str = ""
    tags: List[str] = field(default_factory=list)


class GraceIDE:
    """
    Grace IDE - Live Development Environment

    Provides:
    - Block-card representation for functional elements
    - Visual flow editor with drag-and-drop interface
    - Template library for common patterns
    """

    def __init__(self):
        self.version = "1.0.0"
        self.flows: Dict[str, VisualFlow] = {}
        self.block_registry = self._initialize_block_registry()

        logger.info("Grace IDE initialized")

    def _initialize_block_registry(self) -> Dict[str, Dict[str, Any]]:
        """Initialize the registry of available blocks."""
        return {
            "api_fetch": {
                "name": "API Fetch",
                "description": "Fetch data from REST API endpoint",
                "block_type": BlockType.API_CALL,
                "inputs": [
                    {
                        "name": "url",
                        "data_type": "string",
                        "required": True,
                        "default_value": "",
                        "description": "API endpoint URL",
                    },
                    {
                        "name": "method",
                        "data_type": "string",
                        "required": True,
                        "default_value": "GET",
                        "description": "HTTP method",
                    },
                ],
                "outputs": [
                    {
                        "name": "data",
                        "data_type": "any",
                        "description": "Response data",
                    },
                    {
                        "name": "status_code",
                        "data_type": "int",
                        "description": "HTTP status code",
                    },
                ],
            },
            "sentiment_analysis": {
                "name": "Sentiment Analysis",
                "description": "Analyze sentiment of text data",
                "block_type": BlockType.ANALYSIS,
                "inputs": [
                    {
                        "name": "text_data",
                        "data_type": "any",
                        "required": True,
                        "default_value": None,
                        "description": "Text data for analysis",
                    }
                ],
                "outputs": [
                    {
                        "name": "sentiment_scores",
                        "data_type": "dict",
                        "description": "Sentiment scores",
                    },
                    {
                        "name": "classification",
                        "data_type": "string",
                        "description": "Overall sentiment",
                    },
                ],
            },
            "data_filter": {
                "name": "Data Filter",
                "description": "Filter data based on conditions",
                "block_type": BlockType.TRANSFORMATION,
                "inputs": [
                    {
                        "name": "data",
                        "data_type": "any",
                        "required": True,
                        "default_value": None,
                        "description": "Input data",
                    },
                    {
                        "name": "filter_condition",
                        "data_type": "string",
                        "required": True,
                        "default_value": "",
                        "description": "Filter condition",
                    },
                ],
                "outputs": [
                    {
                        "name": "filtered_data",
                        "data_type": "any",
                        "description": "Filtered data",
                    }
                ],
            },
        }

    def create_flow(self, name: str, description: str, creator_id: str) -> str:
        """Create a new visual flow."""
        flow_id = f"flow_{uuid.uuid4().hex[:8]}"

        flow = VisualFlow(
            flow_id=flow_id,
            name=name,
            description=description,
            blocks=[],
            connections=[],
            creator_id=creator_id,
        )

        self.flows[flow_id] = flow
        logger.info(f"Created flow {flow_id}: {name}")

        return flow_id

    def get_flow(self, flow_id: str) -> Optional[VisualFlow]:
        """Get a flow by ID."""
        return self.flows.get(flow_id)

    def add_block_to_flow(
        self, flow_id: str, block_type_id: str, position: Dict[str, float]
    ) -> str:
        """Add a new block to a flow."""
        if flow_id not in self.flows:
            raise ValueError(f"Flow {flow_id} not found")

        if block_type_id not in self.block_registry:
            raise ValueError(f"Block type {block_type_id} not found")

        block_template = self.block_registry[block_type_id]
        block_id = f"block_{uuid.uuid4().hex[:8]}"

        # Convert dict inputs/outputs to proper objects
        inputs = [BlockInput(**inp) for inp in block_template["inputs"]]
        outputs = [BlockOutput(**out) for out in block_template["outputs"]]

        block = FlowBlock(
            block_id=block_id,
            name=block_template["name"],
            description=block_template["description"],
            block_type=block_template["block_type"],
            inputs=inputs,
            outputs=outputs,
            position=position,
        )

        flow = self.flows[flow_id]
        flow.blocks.append(block)

        logger.info(f"Added block {block_id} to flow {flow_id}")
        return block_id

    def get_block_registry(self) -> Dict[str, Dict[str, Any]]:
        """Get the complete block registry."""
        return self.block_registry

    def get_stats(self) -> Dict[str, Any]:
        """Get IDE statistics."""
        return {
            "flows": {"total": len(self.flows)},
            "blocks": {
                "types_available": len(self.block_registry),
                "total_in_flows": sum(len(flow.blocks) for flow in self.flows.values()),
            },
        }
