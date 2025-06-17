from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field
from langchain_ollama.llms import OllamaLLM
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough

import logging
from langchain_core.utils.function_calling import convert_to_openai_tool
from typing_extensions import Self, is_typeddict
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from operator import itemgetter
from pydantic.v1 import BaseModel as BaseModelV1

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.language_models.base import LanguageModelInput
import re
import json
import uuid

from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    JsonOutputParser,
    PydanticOutputParser,
    PydanticToolsParser,
)
from langchain_core.utils.pydantic import TypeBaseModel, is_basemodel_subclass


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)

class CustomChatModel(BaseChatModel):

    model_name: str = Field(alias="model")
    """The name of the model"""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2
    tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]] = []

    def extract_json_from_code_block(self, text: str) -> dict:
    # Use regex to extract content between ```python and ```
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            return {}

        code_block = match.group(1)

        try:
            return json.loads(code_block)
        except json.JSONDecodeError:
            return {}


    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        tool_instructions = ""
        for tool in self.tools:
            tool_instructions += str(convert_to_openai_tool(tool)) + "\n"

        system_prompt = f"""\
            You are an assistant that has access to the following set of tools. 
            Here are the names and descriptions for each tool:

            {tool_instructions}

            Given the user input, return the name and input of the tool to use. 
            Return your response as a JSON blob with 'name' and 'arguments' keys.

            The `arguments` should be a dictionary, with keys corresponding 
            to the argument names and the values corresponding to the requested values.
            """
        
        messages.insert(0, SystemMessage(content=system_prompt))
        

        all_messages_string = "\n".join(message.content for message in messages)
        logger.info(f"Messages string: {all_messages_string}")
        logger.info(f"Messages: {messages}")
        model = OllamaLLM(model=self.model_name)
        response = model.invoke(messages)
        
        logger.info(f"Response type: {type(response)}")

        tool_call = self.extract_json_from_code_block(response)
        parameters = {}

        if "name" in tool_call and "arguments" in tool_call:
            response = ""
            parameters = {
                "name": tool_call["name"],
                "args": tool_call["arguments"],
                "type": "tool_call",
                "id": f"call_{uuid.uuid4().hex}"
            }
            message = AIMessage(
                content="",
                tool_calls = [parameters],
                additional_kwargs={},  # Used to add additional payload to the message
                response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
                "model_name": self.model_name,
                },
            )
        else:
            message = AIMessage(
                content=response,
                additional_kwargs={},  # Used to add additional payload to the message
                response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
                "model_name": self.model_name,
                },
            )
        logger.info(f"Response: {message}")
        #

        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])
    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function,
                "auto" to automatically determine which function to call
                with the option to not call any function, "any" to enforce that some
                function is called, or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        self.tools = tools
        return super().bind(tools=formatted_tools, **kwargs)
    

    def with_structured_output(
        self,
        schema: Union[dict, type],
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[dict, BaseModel]]:
        _ = kwargs.pop("strict", None)
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": formatted_tool,
                },
            )
            if is_pydantic_schema:
                output_parser: Runnable = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(
                format="json",
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_schema":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
            if is_pydantic_schema:
                schema = cast(TypeBaseModel, schema)
                if issubclass(schema, BaseModelV1):
                    response_format = schema.schema()
                else:
                    response_format = schema.model_json_schema()
                llm = self.bind(
                    format=response_format,
                    ls_structured_output_format={
                        "kwargs": {"method": method},
                        "schema": schema,
                    },
                )
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                if is_typeddict(schema):
                    response_format = convert_to_json_schema(schema)
                    if "required" not in response_format:
                        response_format["required"] = list(
                            response_format["properties"].keys()
                        )
                else:
                    # is JSON schema
                    response_format = cast(dict, schema)
                llm = self.bind(
                    format=response_format,
                    ls_structured_output_format={
                        "kwargs": {"method": method},
                        "schema": response_format,
                    },
                )
                output_parser = JsonOutputParser()
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling', "
                f"'json_schema', or 'json_mode'. Received: '{method}'"
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

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