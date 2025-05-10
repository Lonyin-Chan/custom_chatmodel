from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field
from langchain_ollama.llms import OllamaLLM

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CustomChatModel(BaseChatModel):

    model_name: str = Field(alias="model")
    """The name of the model"""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2


    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        all_messages_string = "".join(message.content for message in messages)
        logger.info(f"Messages: {all_messages_string}")
        model = OllamaLLM(model=self.model_name)
        response = model.invoke(all_messages_string)
        logger.info(f"Response: {response}")
        logger.info(f"Response type: {type(response)}")

        message = AIMessage(
            content=response,
            additional_kwargs={},  # Used to add additional payload to the message
            response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
                "model_name": self.model_name,
            },
        )
        #

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "lonyin-chat-model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }