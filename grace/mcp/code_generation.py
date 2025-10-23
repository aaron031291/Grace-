"""
Grace AI MCP Code Generation Tool - LLM-powered code generation
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CodeGenerationTool:
    """Tool for generating code through MCP."""
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
    
    async def generate_function(self, description: str, language: str = "python") -> Dict[str, Any]:
        """Generate a function based on description."""
        logger.info(f"Generating {language} function: {description}")
        
        prompt = f"""Generate a well-documented {language} function that: {description}
        
Provide only the function code with docstring, no explanation."""
        
        if self.llm_service:
            code = await self.llm_service.generate_text(prompt)
        else:
            code = f"# Generated function for: {description}\ndef generated_function():\n    pass"
        
        return {
            "language": language,
            "description": description,
            "generated_code": code,
            "status": "success"
        }
    
    async def refactor_code(self, code: str, language: str = "python", improvement_type: str = "clarity") -> Dict[str, Any]:
        """Refactor code for improvements."""
        logger.info(f"Refactoring {language} code for {improvement_type}")
        
        prompt = f"""Refactor the following {language} code to improve {improvement_type}:

{code}

Provide only the refactored code, no explanation."""
        
        if self.llm_service:
            refactored = await self.llm_service.generate_text(prompt)
        else:
            refactored = code
        
        return {
            "language": language,
            "improvement_type": improvement_type,
            "original_code": code,
            "refactored_code": refactored,
            "status": "success"
        }

async def code_generation_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for code generation operations through MCP."""
    operation = params.get("operation", "generate_function")
    language = params.get("language", "python")
    
    tool = CodeGenerationTool()
    
    if operation == "generate_function":
        description = params.get("description", "")
        result = await tool.generate_function(description, language)
    elif operation == "refactor":
        code = params.get("code", "")
        improvement_type = params.get("improvement_type", "clarity")
        result = await tool.refactor_code(code, language, improvement_type)
    else:
        result = {"error": f"Unknown operation: {operation}"}
    
    return result
