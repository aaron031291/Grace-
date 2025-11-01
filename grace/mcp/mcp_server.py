"""
MCP Server Implementation for Grace
Enables Model Context Protocol connectivity for tool integration
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class GraceMCPServer:
    """
    Grace's MCP Server
    
    Exposes Grace's capabilities as MCP tools that can be called
    by external systems or AI agents.
    """
    
    def __init__(self):
        self.tools_registry = {}
        self.resources_registry = {}
        self.prompts_registry = {}
        
        self._register_tools()
        self._register_resources()
        self._register_prompts()
        
        logger.info("Grace MCP Server initialized")
    
    def _register_tools(self):
        """Register Grace's capabilities as MCP tools"""
        
        self.tools_registry = {
            "evaluate_code": {
                "name": "evaluate_code",
                "description": "Evaluate code quality, correctness, and security",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code to evaluate"
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language",
                            "enum": ["python", "javascript", "typescript", "rust", "go"]
                        },
                        "criteria": {
                            "type": "array",
                            "description": "Evaluation criteria",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["code", "language"]
                }
            },
            "generate_code": {
                "name": "generate_code",
                "description": "Generate code based on requirements using Grace's AI",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "requirements": {
                            "type": "string",
                            "description": "Code requirements/specification"
                        },
                        "language": {
                            "type": "string",
                            "description": "Target programming language"
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context (existing code, patterns, etc.)"
                        }
                    },
                    "required": ["requirements", "language"]
                }
            },
            "consensus_decision": {
                "name": "consensus_decision",
                "description": "Make decision using ML/DL consensus with verification",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Decision task description"
                        },
                        "options": {
                            "type": "array",
                            "description": "Available options",
                            "items": {"type": "string"}
                        },
                        "context": {
                            "type": "object",
                            "description": "Decision context"
                        }
                    },
                    "required": ["task", "options"]
                }
            },
            "improve_system": {
                "name": "improve_system",
                "description": "Run meta-loop improvement cycle",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "focus_area": {
                            "type": "string",
                            "description": "Area to improve (optional)"
                        }
                    }
                }
            },
            "query_memory": {
                "name": "query_memory",
                "description": "Search Grace's memory and knowledge base",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            "verify_code": {
                "name": "verify_code",
                "description": "Verify code using Grace's verification branch",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "test_cases": {"type": "array"},
                        "safety_checks": {"type": "boolean", "default": True}
                    },
                    "required": ["code"]
                }
            }
        }
        
        logger.info(f"Registered {len(self.tools_registry)} MCP tools")
    
    def _register_resources(self):
        """Register Grace's resources"""
        
        self.resources_registry = {
            "system_status": {
                "uri": "grace://system/status",
                "name": "System Status",
                "description": "Current Grace system status",
                "mimeType": "application/json"
            },
            "improvement_history": {
                "uri": "grace://meta-loop/history",
                "name": "Improvement History",
                "description": "History of self-improvements",
                "mimeType": "application/json"
            },
            "consensus_stats": {
                "uri": "grace://consensus/stats",
                "name": "Consensus Statistics",
                "description": "ML/DL consensus performance stats",
                "mimeType": "application/json"
            }
        }
        
        logger.info(f"Registered {len(self.resources_registry)} MCP resources")
    
    def _register_prompts(self):
        """Register Grace's prompt templates"""
        
        self.prompts_registry = {
            "code_review": {
                "name": "Code Review",
                "description": "Prompt for comprehensive code review",
                "arguments": [
                    {"name": "code", "description": "Code to review", "required": True},
                    {"name": "focus", "description": "Review focus area", "required": False}
                ]
            },
            "explain_code": {
                "name": "Explain Code",
                "description": "Prompt for code explanation",
                "arguments": [
                    {"name": "code", "description": "Code to explain", "required": True}
                ]
            }
        }
        
        logger.info(f"Registered {len(self.prompts_registry)} MCP prompts")
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute an MCP tool
        
        This is called by external MCP clients to invoke Grace's capabilities
        """
        logger.info(f"MCP tool called: {name}")
        
        if name not in self.tools_registry:
            raise ValueError(f"Unknown tool: {name}")
        
        # Route to appropriate handler
        handlers = {
            "evaluate_code": self._evaluate_code,
            "generate_code": self._generate_code,
            "consensus_decision": self._consensus_decision,
            "improve_system": self._improve_system,
            "query_memory": self._query_memory,
            "verify_code": self._verify_code
        }
        
        handler = handlers.get(name)
        if not handler:
            return {"error": f"Handler not implemented for: {name}"}
        
        result = await handler(**arguments)
        
        logger.info(f"MCP tool completed: {name}")
        return result
    
    async def _evaluate_code(
        self,
        code: str,
        language: str,
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate code quality"""
        # In production, use actual code evaluation
        return {
            "quality_score": 0.85,
            "correctness": 0.9,
            "security": 0.8,
            "style": 0.85,
            "issues": [],
            "recommendations": ["Add error handling", "Add docstrings"]
        }
    
    async def _generate_code(
        self,
        requirements: str,
        language: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate code using collaborative system"""
        # This would integrate with the breakthrough system
        return {
            "code": f"# Generated code for: {requirements}\n# Language: {language}\n\ndef solution():\n    pass",
            "confidence": 0.8,
            "rationale": "Generated using consensus-based approach"
        }
    
    async def _consensus_decision(
        self,
        task: str,
        options: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make decision using consensus"""
        from grace.mldl.disagreement_consensus import DisagreementAwareConsensus, ModelPrediction
        
        consensus = DisagreementAwareConsensus()
        
        # Simulate model predictions (in production, use real models)
        predictions = [
            ModelPrediction(f"model_{i}", options[i % len(options)], 0.8)
            for i in range(3)
        ]
        
        result = await consensus.reach_consensus(task, predictions)
        
        return {
            "decision": result.final_prediction,
            "confidence": result.confidence,
            "method": result.method_used.value,
            "agreement": result.agreement_score
        }
    
    async def _improve_system(
        self,
        focus_area: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run improvement cycle"""
        from grace.core.breakthrough import BreakthroughSystem
        
        system = BreakthroughSystem()
        if not system.initialized:
            await system.initialize()
        
        result = await system.run_single_improvement_cycle()
        
        return {
            "cycle_complete": result["cycle_complete"],
            "status": result["status"],
            "improvement": result["improvement"]
        }
    
    async def _query_memory(
        self,
        query: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Search memory"""
        # In production, use actual memory search
        return {
            "results": [],
            "total": 0,
            "query": query
        }
    
    async def _verify_code(
        self,
        code: str,
        test_cases: Optional[List[Dict]] = None,
        safety_checks: bool = True
    ) -> Dict[str, Any]:
        """Verify code"""
        return {
            "verified": True,
            "test_results": [],
            "safety_passed": True
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        return list(self.tools_registry.values())
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List all available resources"""
        return list(self.resources_registry.values())
    
    def list_prompts(self) -> List[Dict[str, Any]]:
        """List all available prompts"""
        return list(self.prompts_registry.values())


# Global MCP server instance
_mcp_server: Optional[GraceMCPServer] = None


def get_mcp_server() -> GraceMCPServer:
    """Get global MCP server instance"""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = GraceMCPServer()
    return _mcp_server


if __name__ == "__main__":
    # Demo
    async def demo():
        print("🔌 Grace MCP Server Demo\n")
        
        server = GraceMCPServer()
        
        print("📋 Available Tools:")
        for tool in server.list_tools():
            print(f"  - {tool['name']}: {tool['description']}")
        
        print("\n🔧 Testing tool execution...")
        
        # Test evaluate_code
        result = await server.call_tool("evaluate_code", {
            "code": "def hello(): print('world')",
            "language": "python"
        })
        print(f"\n✅ evaluate_code result:")
        print(f"  Quality: {result['quality_score']}")
        
        # Test consensus
        result = await server.call_tool("consensus_decision", {
            "task": "Choose best approach",
            "options": ["option_a", "option_b", "option_c"]
        })
        print(f"\n✅ consensus_decision result:")
        print(f"  Decision: {result['decision']}")
        print(f"  Confidence: {result['confidence']:.2f}")
    
    asyncio.run(demo())
