"""
Transformer Specialist - Attention-based NLP

Use cases:
- Policy document analysis
- Governance text understanding
- Semantic similarity
- Text classification
- Question answering
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .base_deep_specialist import BaseDeepLearningSpecialist
from ..base_specialist import SpecialistCapability

logger = logging.getLogger(__name__)


class TransformerSpecialist(BaseDeepLearningSpecialist):
    """
    Transformer Specialist using Hugging Face pretrained models.
    
    Use cases:
    - Policy document classification
    - Governance compliance checking
    - Semantic similarity for audit logs
    - Text summarization
    - Question answering
    - Named entity recognition
    
    Features:
    - Pretrained BERT, RoBERTa, DistilBERT
    - Fine-tuning on custom data
    - GPU acceleration
    - Batch processing
    """
    
    def __init__(
        self,
        specialist_id: str = "transformer_specialist",
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "Transformers not available. Install with: "
                "pip install transformers"
            )
        
        capabilities = [
            SpecialistCapability.CLASSIFICATION,
            SpecialistCapability.PATTERN_RECOGNITION
        ]
        super().__init__(specialist_id, capabilities, device)
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info(f"Initialized TransformerSpecialist with {model_name}")
    
    def build_model(self, input_shape: Tuple, **kwargs) -> nn.Module:
        """Build transformer model"""
        self.num_labels = kwargs.get('num_labels', self.num_labels)
        
        # Load pretrained model for classification
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        # Use CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Built Transformer model: {self.model_name}, "
                   f"num_labels={self.num_labels}")
        
        return model
    
    def tokenize(self, texts: List[str]) -> Dict[str, 'torch.Tensor']:
        """Tokenize texts"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def predict_text(self, text: str) -> Tuple[int, float, np.ndarray]:
        """
        Predict class for text
        
        Args:
            text: Input text
        
        Returns:
            predicted_class, confidence, class_probabilities
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained/loaded")
        
        # Tokenize
        inputs = self.tokenize([text])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        class_probs = probabilities[0].cpu().numpy()
        
        return predicted_class, confidence, class_probs
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get text embeddings (for semantic similarity)
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector
        """
        # Use base model for embeddings
        if not hasattr(self, 'embedding_model'):
            self.embedding_model = AutoModel.from_pretrained(self.model_name)
            self.embedding_model = self.embedding_model.to(self.device)
        
        # Tokenize
        inputs = self.tokenize([text])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        self.embedding_model.eval()
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings.cpu().numpy().flatten()
    
    def _process_output(self, output) -> Tuple[Any, float]:
        """Process transformer output"""
        # Output is model output object
        logits = output.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence
    
    async def predict_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> 'SpecialistPrediction':
        """
        Make text classification prediction
        
        Expected input_data:
        {
            'text': str,
            'task': 'classify' or 'embed' (optional)
        }
        """
        if not self.is_trained and 'text' not in input_data:
            from ..base_specialist import SpecialistPrediction
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': 'Model not trained or no text provided'}
            )
        
        try:
            text = input_data.get('text', '')
            task = input_data.get('task', 'classify')
            
            if task == 'embed':
                # Return embeddings
                embeddings = self.get_embeddings(text)
                
                from ..base_specialist import SpecialistPrediction
                return SpecialistPrediction(
                    specialist_id=self.specialist_id,
                    prediction_value=embeddings.tolist(),
                    confidence=1.0,
                    metadata={
                        'task': 'embedding',
                        'model': self.model_name,
                        'embedding_dim': len(embeddings)
                    }
                )
            else:
                # Classification
                predicted_class, confidence, class_probs = self.predict_text(text)
                
                from ..base_specialist import SpecialistPrediction
                return SpecialistPrediction(
                    specialist_id=self.specialist_id,
                    prediction_value=int(predicted_class),
                    confidence=float(confidence),
                    metadata={
                        'task': 'classification',
                        'model': self.model_name,
                        'class_probabilities': class_probs.tolist()
                    }
                )
        
        except Exception as e:
            logger.error(f"Transformer prediction failed: {e}")
            from ..base_specialist import SpecialistPrediction
            return SpecialistPrediction(
                specialist_id=self.specialist_id,
                prediction_value=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
