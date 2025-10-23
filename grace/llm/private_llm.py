"""
Private LLM - Local model inference without external APIs
"""

from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Available local LLM providers"""
    LLAMA_CPP = "llama_cpp"      # llama.cpp backend
    TRANSFORMERS = "transformers" # HuggingFace transformers
    VLLM = "vllm"                # vLLM for production
    OLLAMA = "ollama"            # Ollama local deployment
    ONNX = "onnx"                # ONNX Runtime


@dataclass
class ModelConfig:
    """Configuration for a local LLM model"""
    name: str
    provider: LLMProvider
    model_path: str
    context_length: int = 2048
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    gpu_layers: int = 0  # Number of layers on GPU
    threads: int = 4
    batch_size: int = 8
    streaming: bool = False


class PrivateLLM:
    """
    Private LLM Manager - Multi-model local inference
    
    Supports:
    - Multiple model backends (llama.cpp, transformers, vLLM, Ollama)
    - GPU acceleration (CUDA, Metal, ROCm)
    - CPU-only fallback
    - Quantized models (4-bit, 8-bit)
    - Streaming responses
    - Batched inference
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialize_backend()
        
        logger.info(f"PrivateLLM initialized: {config.name} ({config.provider.value})")
    
    def _initialize_backend(self):
        """Initialize the appropriate backend"""
        if self.config.provider == LLMProvider.LLAMA_CPP:
            self._init_llama_cpp()
        elif self.config.provider == LLMProvider.TRANSFORMERS:
            self._init_transformers()
        elif self.config.provider == LLMProvider.VLLM:
            self._init_vllm()
        elif self.config.provider == LLMProvider.OLLAMA:
            self._init_ollama()
        elif self.config.provider == LLMProvider.ONNX:
            self._init_onnx()
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def _init_llama_cpp(self):
        """Initialize llama.cpp backend"""
        try:
            from llama_cpp import Llama
            
            self.model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.context_length,
                n_gpu_layers=self.config.gpu_layers,
                n_threads=self.config.threads,
                n_batch=self.config.batch_size,
                verbose=False
            )
            
            logger.info(f"Loaded llama.cpp model: {self.config.model_path}")
            
        except ImportError:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            raise
        except Exception as e:
            logger.error(f"Failed to load llama.cpp model: {e}")
            raise
    
    def _init_transformers(self):
        """Initialize HuggingFace transformers backend"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Detect device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            # Load model with quantization if available
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            logger.info(f"Loaded transformers model: {self.config.model_path} on {device}")
            
        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load transformers model: {e}")
            raise
    
    def _init_vllm(self):
        """Initialize vLLM backend (production-grade)"""
        try:
            from vllm import LLM, SamplingParams
            
            self.model = LLM(
                model=self.config.model_path,
                max_model_len=self.config.context_length,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=1
            )
            
            logger.info(f"Loaded vLLM model: {self.config.model_path}")
            
        except ImportError:
            logger.error("vllm not installed. Install with: pip install vllm")
            raise
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            raise
    
    def _init_ollama(self):
        """Initialize Ollama backend"""
        try:
            import requests
            
            # Check Ollama is running
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                raise RuntimeError("Ollama server not running")
            
            self.model = "ollama"  # Placeholder
            
            logger.info(f"Connected to Ollama: {self.config.model_path}")
            
        except ImportError:
            logger.error("requests not installed. Install with: pip install requests")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise
    
    def _init_onnx(self):
        """Initialize ONNX Runtime backend"""
        try:
            import onnxruntime as ort
            
            # Create inference session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.model = ort.InferenceSession(self.config.model_path, providers=providers)
            
            logger.info(f"Loaded ONNX model: {self.config.model_path}")
            
        except ImportError:
            logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
            raise
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text from prompt
        
        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            stream: Enable streaming
            
        Returns:
            Generated text and metadata
        """
        # Use config defaults if not specified
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        
        if self.config.provider == LLMProvider.LLAMA_CPP:
            return self._generate_llama_cpp(prompt, max_tokens, temperature, top_p, stop, stream)
        
        elif self.config.provider == LLMProvider.TRANSFORMERS:
            return self._generate_transformers(prompt, max_tokens, temperature, top_p, stop)
        
        elif self.config.provider == LLMProvider.VLLM:
            return self._generate_vllm(prompt, max_tokens, temperature, top_p, stop)
        
        elif self.config.provider == LLMProvider.OLLAMA:
            return self._generate_ollama(prompt, max_tokens, temperature, top_p, stop, stream)
        
        elif self.config.provider == LLMProvider.ONNX:
            return self._generate_onnx(prompt, max_tokens, temperature, top_p)
        
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def _generate_llama_cpp(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        stream: bool
    ) -> Dict[str, Any]:
        """Generate using llama.cpp"""
        result = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            echo=False,
            stream=stream
        )
        
        if stream:
            return {"stream": result, "provider": "llama_cpp"}
        
        return {
            "text": result["choices"][0]["text"],
            "tokens": result["usage"]["completion_tokens"],
            "provider": "llama_cpp",
            "finish_reason": result["choices"][0]["finish_reason"]
        }
    
    def _generate_transformers(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate using transformers"""
        import torch
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from output
        generated_text = generated_text[len(prompt):].strip()
        
        return {
            "text": generated_text,
            "tokens": len(outputs[0]) - len(inputs["input_ids"][0]),
            "provider": "transformers",
            "finish_reason": "stop"
        }
    
    def _generate_vllm(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate using vLLM"""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )
        
        outputs = self.model.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]
        
        return {
            "text": output.text,
            "tokens": len(output.token_ids),
            "provider": "vllm",
            "finish_reason": output.finish_reason
        }
    
    def _generate_ollama(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        stream: bool
    ) -> Dict[str, Any]:
        """Generate using Ollama"""
        import requests
        
        payload = {
            "model": self.config.model_path,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop or []
            }
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        
        if stream:
            return {"stream": response.iter_lines(), "provider": "ollama"}
        
        result = response.json()
        
        return {
            "text": result["response"],
            "tokens": result.get("eval_count", 0),
            "provider": "ollama",
            "finish_reason": "stop"
        }
    
    def _generate_onnx(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> Dict[str, Any]:
        """Generate using ONNX Runtime"""
        # ONNX implementation would require custom tokenization and decoding
        # This is a placeholder - actual implementation depends on model format
        
        logger.warning("ONNX generation not fully implemented")
        
        return {
            "text": "[ONNX generation placeholder]",
            "tokens": 0,
            "provider": "onnx",
            "finish_reason": "placeholder"
        }
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Chat completion interface
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Enable streaming
            
        Returns:
            Generated response
        """
        # Format messages into prompt
        prompt = self._format_chat_prompt(messages)
        
        # Generate
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream
        )
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string"""
        # Default formatting - can be customized per model
        formatted = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        formatted.append("Assistant:")  # Prompt for next response
        
        return "\n\n".join(formatted)
