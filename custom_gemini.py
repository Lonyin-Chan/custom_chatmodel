import json
from typing import Any, Dict, List, Optional, Union, Type

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel, generate_from_stream
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool, tool

# --- Mock Implementation of Your Company's Gemini Package ---
# This function simulates the behavior of your internal package.
# It's designed to respond similarly to how Gemini would, allowing
# this example to be fully runnable.
#
# Replace the contents of this function with the actual call
# to your company's package.

def prompt_gemini(
    system_instructions: str,
    prompt: str,
    response_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """
    A mock function that simulates calling your internal Gemini API.

    Args:
        system_instructions: The system prompt for the model.
        prompt: The main user prompt and conversation history.
        response_schema: A JSON schema for function calling or structured output.

    Returns:
        A string response from the simulated model.
    """
    print("--- Calling Mock Gemini API ---")
    print(f"System Instructions:\n{system_instructions}")
    print(f"Prompt:\n{prompt}")
    print(f"Response Schema:\n{json.dumps(response_schema, indent=2) if response_schema else 'None'}")
    print("-----------------------------\n")

    # 1. Simulation for Structured Output
    # If a schema is provided and it's not for tools, return a matching JSON string.
    if response_schema and response_schema.get("name") != "tool_calls":
        if response_schema.get("name") == "WeatherSearch":
            return json.dumps({
                "location": "Boston, MA",
                "unit": "celsius"
            })
        # Generic fallback for other schemas
        return json.dumps({
            "response": "This is a structured response based on the provided schema."
        })

    # 2. Simulation for Tool Calling
    # If the prompt mentions a tool and a tool schema is present, simulate a tool call.
    if "weather" in prompt.lower() and response_schema and response_schema.get("name") == "tool_calls":
        # Gemini responds with a specific JSON structure for tool calls.
        # We simulate that structure here.
        tool_call_response = {
            "tool_calls": [
                {
                    "name": "get_weather",
                    "args": {
                        "location": "San Francisco",
                        "unit": "fahrenheit"
                    }
                }
            ]
        }
        return json.dumps(tool_call_response)

    # 3. Simulation for a standard chat response
    return "Hello! This is a mock response from the custom Gemini model. How can I assist you today?"


# --- Custom LangChain Chat Model Definition ---

class CustomGeminiChatModel(BaseChatModel):
    """
    A custom LangChain Chat Model that uses a proprietary internal package
    to interact with Google's Gemini models.
    """

    @property
    def _llm_type(self) -> str:
        """A mandatory property to identify the LLM type."""
        return "custom-gemini-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        The core logic for interacting with the LLM. It formats the prompt,
        handles tools and structured output, calls the API, and parses the response.
        """
        # 1. Extract system instructions and format the main prompt.
        system_instructions, formatted_prompt = self._format_messages(messages)

        # 2. Prepare the response schema for tools or structured output.
        # This is the key part for enabling tool use and .with_structured_output().
        response_schema = self._prepare_response_schema(**kwargs)

        # 3. Make the API call using your internal package.
        # This is where you would replace the mock function with your real one.
        response_str = prompt_gemini(
            system_instructions=system_instructions,
            prompt=formatted_prompt,
            response_schema=response_schema
        )

        # 4. Parse the string response into a LangChain AIMessage.
        generation = self._parse_response(response_str, response_schema)

        return ChatResult(generations=[generation])

    @staticmethod
    def _format_messages(messages: List[BaseMessage]) -> (str, str):
        """
        Converts a list of LangChain messages into a system prompt string
        and a single conversational prompt string.
        """
        system_instructions_list = []
        prompt_list = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_instructions_list.append(msg.content)
            elif isinstance(msg, HumanMessage):
                prompt_list.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                # For AI messages with tool calls, we format them clearly.
                if msg.tool_calls:
                    tool_calls_str = json.dumps([
                        {"id": tc["id"], "name": tc["name"], "args": tc["args"]} for tc in msg.tool_calls
                    ], indent=2)
                    prompt_list.append(f"AI: (Tool Call)\n{tool_calls_str}")
                else:
                    prompt_list.append(f"AI: {msg.content}")
            elif isinstance(msg, ToolMessage):
                prompt_list.append(f"Tool (id: {msg.tool_call_id}):\nResult: {msg.content}")

        system_instructions = "\n".join(system_instructions_list)
        formatted_prompt = "\n\n".join(prompt_list)
        return system_instructions, formatted_prompt

    def _prepare_response_schema(self, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """
        Checks for bound tools or a structured output schema in the kwargs
        and formats it for the Gemini API.
        """
        # Case 1: Tools are bound using .bind_tools()
        if "tools" in kwargs:
            tools = kwargs["tools"]
            return {
                "name": "tool_calls",
                "description": "The user wants you to use a tool.",
                "properties": {
                    tool.name: tool.description for tool in tools
                },
                "type": "object",
                "required": [tool.name for tool in tools],
                "definitions": {
                    tool.name: tool.args_schema.schema() for tool in tools
                }
            }

        # Case 2: Structured output is requested via .with_structured_output()
        # LangChain passes the schema in the 'functions' or 'function_call' kwarg.
        # We check for a 'response_format' that contains the schema.
        if "response_format" in kwargs and "schema" in kwargs["response_format"]:
            return kwargs["response_format"]["schema"]
            
        return None

    @staticmethod
    def _parse_response(response_str: str, response_schema: Optional[Dict[str, Any]]) -> ChatGeneration:
        """
        Parses the raw string from the API into an AIMessage,
        detecting tool calls or structured output.
        """
        try:
            # Assume the response might be JSON (for tool calls or structured output)
            response_json = json.loads(response_str)
            
            # Check for Gemini's tool calling format
            if "tool_calls" in response_json and isinstance(response_json["tool_calls"], list):
                tool_calls = [
                    {
                        "name": call["name"],
                        "args": call["args"],
                        "id": f"call_{call['name']}_{i}", # Generate a unique ID for the call
                    }
                    for i, call in enumerate(response_json["tool_calls"])
                ]
                message = AIMessage(content="", tool_calls=tool_calls)
            else:
                # If it's JSON but not a tool call, it's structured output.
                # We return it as a string, and LangChain will handle the final parsing.
                message = AIMessage(content=response_str)
        except json.JSONDecodeError:
            # If it's not JSON, it's a standard text response.
            message = AIMessage(content=response_str)

        return ChatGeneration(message=message)
        
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Streaming is not implemented in this example as the mock function
        # returns a single string. To implement this, you would need your
        # internal package to support streaming responses.
        yield self._generate(messages, stop, run_manager, **kwargs).generations[0]


# --- Usage Examples ---

if __name__ == "__main__":
    # Initialize the custom model
    custom_model = CustomGeminiChatModel()

    # --- Example 1: Standard Chat Invocation ---
    print("--- 1. Standard Chat Invocation ---")
    messages = [HumanMessage(content="What is the weather like today?")]
    result = custom_model.invoke(messages)
    print("Model Response:")
    print(result.content)
    print("-" * 30, "\n")


    # --- Example 2: Tool Calling ---
    print("--- 2. Tool Calling Invocation ---")

    # Define a simple tool
    @tool
    def get_weather(location: str, unit: str = "celsius") -> str:
        """Gets the current weather in a given location."""
        return f"The weather in {location} is 25 degrees {unit}."

    # Bind the tool to the model
    model_with_tools = custom_model.bind_tools([get_weather])

    # Invoke the model with a prompt that should trigger the tool
    messages = [HumanMessage(content="What's the weather like in San Francisco?")]
    ai_msg_with_tool_call = model_with_tools.invoke(messages)

    print("Model Response (Tool Call):")
    print(ai_msg_with_tool_call)

    # Simulate running the tool and returning the result to the model
    if ai_msg_with_tool_call.tool_calls:
        tool_call = ai_msg_with_tool_call.tool_calls[0]
        tool_output = get_weather.invoke(tool_call["args"])

        print("\nTool Output:")
        print(tool_output)

        # Create a new list of messages including the tool call and its output
        messages.append(ai_msg_with_tool_call)
        messages.append(
            ToolMessage(content=tool_output, tool_call_id=tool_call["id"])
        )
        
        print("\n--- Calling model again with tool result ---")
        final_response = model_with_tools.invoke(messages)
        print("Final Model Response:")
        print(final_response.content)

    print("-" * 30, "\n")


    # --- Example 3: Structured Output ---
    print("--- 3. Structured Output Invocation ---")

    # Define a Pydantic schema for the desired output
    class WeatherSearch(BaseModel):
        """A structured representation of a weather search query."""
        location: str = Field(..., description="The city and state, e.g., San Francisco, CA")
        unit: Literal["celsius", "fahrenheit"] = Field(..., description="The temperature unit")

    # Create a new chain with the structured output schema
    structured_model = custom_model.with_structured_output(WeatherSearch)

    # Invoke the model
    structured_result = structured_model.invoke("How is the weather in Boston in celsius?")
    
    print("Structured Output Response:")
    print(structured_result)
    print(f"Type of result: {type(structured_result)}")
    print(f"Location: {structured_result.location}")

