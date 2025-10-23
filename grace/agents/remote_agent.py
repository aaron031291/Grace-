"""
Grace AI Remote Agent - Sandboxed interface to execute commands and browse web
"""
import logging
import subprocess
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RemoteAgent:
    """Provides Grace with sandboxed ability to execute commands and browse web."""
    
    def __init__(self, sandbox_dir: str = "/tmp/grace_sandbox"):
        self.sandbox_dir = sandbox_dir
    
    async def execute_command(
        self,
        command: str,
        correlation_id: str = None
    ) -> Dict[str, Any]:
        """Execute a shell command in the sandbox."""
        logger.info(f"Executing command in sandbox: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = {
                "command": command,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "correlation_id": correlation_id
            }
            
            logger.info(f"Command finished with return code: {result.returncode}")
            return output
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {command}")
            return {
                "command": command,
                "return_code": -1,
                "error": "Command timed out",
                "correlation_id": correlation_id
            }
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            return {
                "command": command,
                "return_code": -1,
                "error": str(e),
                "correlation_id": correlation_id
            }
    
    async def browse_web(
        self,
        url: str,
        correlation_id: str = None
    ) -> Dict[str, Any]:
        """Browse a URL and retrieve content."""
        logger.info(f"Browsing web: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            
            return {
                "url": url,
                "status_code": response.status_code,
                "content_length": len(response.content),
                "content_preview": response.text[:500],
                "correlation_id": correlation_id
            }
            
        except Exception as e:
            logger.error(f"Error browsing {url}: {str(e)}")
            return {
                "url": url,
                "error": str(e),
                "correlation_id": correlation_id
            }
