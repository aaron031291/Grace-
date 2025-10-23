"""
Grace AI LLM Service - Interface to local Large Language Models
"""
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class LLMService:
    """Interface to a local LLM (e.g., Ollama) for reasoning and code generation."""
    
    def __init__(self, llm_url: str = "http://localhost:11434", model: str = "mistral"):
        self.llm_url = llm_url
        self.model = model
    
    async def generate_text(self, prompt: str) -> str:
        """Generate text using the LLM."""
        logger.info(f"LLM: Generating text with prompt length {len(prompt)}")
        
        try:
            response = requests.post(
                f"{self.llm_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "")
                logger.info(f"LLM: Generated response of length {len(result)}")
                return result
            else:
                logger.error(f"LLM error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            return ""
    
    async def generate_code(self, problem_description: str, context: str = "") -> str:
        """Generate code to solve a specific problem."""
        prompt = f"""You are an expert Python developer. Generate clean, well-documented code to solve the following problem:

Problem: {problem_description}

Context: {context}

Provide only the Python code, with no explanation."""
        
        return await self.generate_text(prompt)
    
    async def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code for issues and provide suggestions."""
        prompt = f"""You are an expert code reviewer. Analyze the following Python code and identify:
1. Any bugs or potential issues
2. Style or clarity problems
3. Performance improvements
4. Security concerns

Code:
{code}

Provide your analysis in a structured format."""
        
        analysis = await self.generate_text(prompt)
        return {
            "analysis": analysis,
            "code": code
        }
