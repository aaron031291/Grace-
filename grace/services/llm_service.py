"""
Grace AI LLM Service - Interface to local Large Language Models
"""
import logging
from openai import OpenAI, APIError

logger = logging.getLogger(__name__)

class LLMService:
    """
    A service for interacting with Large Language Models.
    Currently supports OpenAI.
    """
    def __init__(self, api_key: str = None):
        """
        Initializes the LLMService.

        Args:
            api_key (str, optional): The OpenAI API key. If not provided,
                                     the service will run in a disabled (dummy) mode.
        """
        if not api_key:
            logger.warning("OpenAI API key not provided. LLMService will run in dummy mode.")
            self.client = None
            self.enabled = False
        else:
            try:
                self.client = OpenAI(api_key=api_key)
                self.enabled = True
                logger.info("LLMService initialized with OpenAI client.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
                self.enabled = False

    async def generate_response(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1500,
        temperature: float = 0.7,
    ) -> str:
        """
        Generates a response from the language model.

        Args:
            prompt (str): The input prompt for the model.
            model (str): The model to use for generation.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The sampling temperature.

        Returns:
            str: The generated text response, or a dummy response if disabled/failed.
        """
        if not self.enabled or not self.client:
            logger.debug("LLMService is disabled. Returning dummy response.")
            return "LLM service is not configured. This is a placeholder response."

        try:
            # In the new OpenAI library, async calls are the default for the base client
            # but are made on the specific endpoints.
            # The library handles the async operations internally when used in an async context.
            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response_text = completion.choices[0].message.content
            logger.info("Successfully generated LLM response.")
            return response_text.strip() if response_text else ""
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error: Could not get a response from the LLM. API Error: {e}"
        except Exception as e:
            logger.error(f"An unexpected error occurred in LLMService: {e}")
            return f"Error: An unexpected error occurred. {e}"

    def is_enabled(self) -> bool:
        """Returns True if the service is configured and enabled, False otherwise."""
        return self.enabled
