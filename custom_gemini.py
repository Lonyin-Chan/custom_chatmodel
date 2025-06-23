import json
from typing import Any, List, Optional, Dict, Union

# LangChain Core Imports
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,I am creating a custom langchain chatmodel from the BaseChatModel class. As part of this, I need to create the _generate() method which I want it to do things like tool calling. Now I am using Gemini 2.0 flash to prompt the LLM but I am not allowed to use google packages to generate content due to GenerateContent vertex aimy company having created it's own package to handle things like governance. So I am only able to prompt Gemini 2.0 flash with one long string that needs to include everything like the tools, system_instructions and content and I will only receive 1 long string as the response. this custom package does allow me to set a response schema though. Also I can create objects and so on using google packages, just not generate content using google packages
    ToolMessage,I am creating a custom langchain chatmodel from the BaseChatModel class. As part of this, I need to create the _generate() method which I want it to do things like tool calling. Now I am using Gemini 2.0 flash to prompt the LLM but I am not allowed to use google packages to generate content due to GenerateContent vertex aimy company having created it's own package to handle things like governance. So I am only able to prompt Gemini 2.0 flash with one long string that needs to include everything like the tools, system_instructions and content and I will only receive 1 long string as the response. this custom package does allow me to set a response schema though. Also I can create objects and so on using google packages, just not generate content using google packages
)
from langchain_core.outputs import ChatGeneration, ChatResult, Generation
from langchain_core.outputs.chat_generation import ChatGenerationChunk
from langchain_core.outputs.llm_output import LLMOutput
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
    format_tool_to_openai_function,
)


# --- Placeholder for your Company's Custom Package ---
# This function simulates the behavior you described: it takes a single string
# and a response schema, and returns a single JSON string.

def my_company_llm_caller(prompt: str, response_schema: Dict[str, Any]) -> str:
    """
    A placeholder for your company's actual LLM calling function.
    It demonstrates how Gemini might respond based on the prompt and schema.
    In a real scenario, this would make an API call.
    """
    print("--- Sending to Custom LLM Package ---")
    print(f"Formatted Prompt:\n{prompt}")
    print(f"\nResponse Schema:\n{json.dumps(response_schema, indent=2)}")
    print("------------------------------------")

    # Simulate Gemini's logic based on the prompt content
    if "weather in Boston" in prompt:
        # If the prompt asks for a tool, simulate Gemini returning a tool call
        # that conforms to the response_schema.
        response_dict = {
            "tool_calls": [
                {
                    "name": "get_weather",
                    "args": {
                        "location": "Boston, MA"
                    }
                }
            ]
        }
        return json.dumps(response_dict)
    else:
        # Otherwise, simulate a standard text response.
        response_dict = {
            "response_text": "Hello! How can I help you today?"
        }
        return json.dumps(response_dict)

# --- Custom Chat Model Implementation ---

class CustomGeminiChatModel(BaseChatModel):
    """
    A custom LangChain ChatModel that uses a proprietary package to call
    the Gemini API with tool-calling capabilities.

    It works by:
    1.  Formatting all messages and tool definitions into a single string prompt.
    2.  Defining a JSON schema that tells Gemini to respond with either a text
        reply or a structured tool call.
    3.  Calling the custom package with the formatted prompt and schema.
    4.  Parsing the JSON string response to create a standard LangChain AIMessage,
        which can include `tool_calls`.
    """
    custom_llm_caller: callable
    model_name: str = "gemini-2.0-flash-custom"

    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "custom-gemini-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        The core logic for the chat model.
        """
        # Step 1: Format the prompt and define the response schema
        tools = kwargs.get("tools", [])
        formatted_prompt = self._format_prompt(messages, tools)
        response_schema = self._define_response_schema(tools)

        # Step 2: Call the custom LLM package
        response_str = self.custom_llm_caller(
            prompt=formatted_prompt,
            response_schema=response_schema
        )

        # Step 3: Parse the response and create the AIMessage
        try:
            response_data = json.loads(response_str)
            if "tool_calls" in response_data and response_data["tool_calls"]:
                # The model decided to call one or more tools
                tool_calls = [
                    {
                        "id": f"call_{i}",
                        "name": call["name"],
                        "args": call["args"],
                        "type": "tool_call", # Standard field
                    }
                    for i, call in enumerate(response_data["tool_calls"])
                ]
                ai_message = AIMessage(
                    content="",  # No text content when calling tools
                    tool_calls=tool_calls
                )
            else:
                # The model returned a standard text response
                text_response = response_data.get("response_text", "")
                ai_message = AIMessage(content=text_response)

        except (json.JSONDecodeError, KeyError):
            # If parsing fails or the structure is wrong, treat the whole
            # output as a simple text response.
            ai_message = AIMessage(content=response_str)

        # Step 4: Wrap the message in the standard LangChain output format
        chat_generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[chat_generation])

    def _format_prompt(self, messages: List[BaseMessage], tools: List[Dict]) -> str:
        """
        Converts the list of messages and tools into a single string.
        """
        prompt_lines = []

        # Add System Message if present
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_lines.append("SYSTEM INSTRUCTIONS:\n" + msg.content)

        # Add Chat History
        prompt_lines.append("\nCHAT HISTORY:")
        for msg in messages:
            if isinstance(msg, HumanMessage):
                prompt_lines.append(f"USER: {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    calls = ', '.join([f"{tc['name']}({json.dumps(tc['args'])})" for tc in msg.tool_calls])
                    prompt_lines.append(f"ASSISTANT (Tool Call): {calls}")
                else:
                    prompt_lines.append(f"ASSISTANT: {msg.content}")
            elif isinstance(msg, ToolMessage):
                prompt_lines.append(f"TOOL_OUTPUT (for {msg.tool_call_id}):\n{msg.content}")


        # Add Tool Definitions
        if tools:
            prompt_lines.append("\nAVAILABLE TOOLS:")
            prompt_lines.append("You can call one or more of the following tools. Respond using the JSON schema provided.")
            for tool in tools:
                # The 'function' key is standard from convert_to_openai_tool
                func = tool.get('function', {})
                name = func.get('name', 'unknown_tool')
                desc = func.get('description', 'No description.')
                params = func.get('parameters', {})
                prompt_lines.append(f"\n- Tool: `{name}`")
                prompt_lines.append(f"  Description: {desc}")
                prompt_lines.append(f"  Parameters Schema: {json.dumps(params)}")

        prompt_lines.append("\nYOUR TASK: Based on the instructions, history, and available tools, provide your response. If a tool is appropriate, use it. Otherwise, respond to the user directly.")
        return "\n".join(prompt_lines)

    def _define_response_schema(self, tools: List[Dict]) -> Dict[str, Any]:
        """
        Creates the JSON schema for the expected response from Gemini.
        """
        tool_names = [tool.get('function', {}).get('name') for tool in tools]

        schema = {
            "type": "object",
            "properties": {
                "response_text": {
                    "type": "string",
                    "description": "The natural language response to the user, if no tool is being called."
                },
                "tool_calls": {
                    "type": "array",
                    "description": "A list of tool calls to make.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the tool to call.",
                                # Restrict the name to available tools for better accuracy
                                "enum": tool_names if tool_names else [],
                            },
                            "args": {
                                "type": "object",
                                "description": "The arguments for the tool, as a key-value map.",
                                "properties": {} # In a real scenario, you could dynamically build this from tool schemas
                            }
                        },
                        "required": ["name", "args"]
                    }
                }
            }
        }
        return schema


# --- Example Usage ---

# 1. Define a tool the model can use (using Pydantic is standard)
class GetWeather(BaseModel):
    """Gets the current weather in a given location."""
    location: str = Field(..., description="The city and state, e.g., San Francisco, CA")

# 2. Instantiate the custom chat model
#    We pass our placeholder function `my_company_llm_caller` here.
#    You would pass your actual company's function.
custom_chat_model = CustomGeminiChatModel(custom_llm_caller=my_company_llm_caller)

# 3. Bind the tool to the model
#    This is the standard LangChain way to make tools available to a model.
#    Our custom model knows how to handle this from the `kwargs`.
model_with_tools = custom_chat_model.bind_tools([GetWeather])

# 4. Invoke the model with a prompt that should trigger the tool
prompt = "What is the weather in Boston?"
result = model_with_tools.invoke(prompt)

print("\n--- LangChain Final Output ---")
print(result)
print("----------------------------\n")

# Verify that the output is a standard AIMessage with tool_calls
assert isinstance(result, AIMessage)
assert result.content == ""
assert len(result.tool_calls) == 1
assert result.tool_calls[0]['name'] == 'get_weather'
assert result.tool_calls[0]['args'] == {'location': 'Boston, MA'}

print("✅ Successfully created and invoked the custom model with tool calling!")

# Example of a regular chat interaction
print("\n--- Invoking without a tool call ---")
result_text = model_with_tools.invoke("Hi there!")
print("\n--- LangChain Final Output ---")
print(result_text)
print("----------------------------\n")
assert result_text.content != ""
assert not result_text.tool_calls
print("✅ Successfully handled a non-tool-calling interaction.")

