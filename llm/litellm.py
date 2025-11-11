"""LiteLLM client helper using OpenAI-compatible API."""

from __future__ import annotations

import json
from typing import List, Optional

import requests


class LiteLLMClient:
    def __init__(
        self,
        url: str,
        model: str,
        api_key: str,
        stream: bool = True,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize LiteLLM client.
        
        Args:
            url: Base URL for the LiteLLM API (e.g., "https://api.openai.com/v1/chat/completions")
            model: Model name to use
            api_key: API key for authentication
            stream: Whether to stream responses
            system_prompt: Optional system prompt to prepend to conversations
            max_tokens: Maximum tokens in response (default: 2048)
        """
        self.url = url
        self.model = model
        self.api_key = api_key
        self.stream = stream
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def _build_messages(self, user_text: str) -> List[dict]:
        messages: List[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_text})
        return messages

    def query(self, user_text: str) -> str:
        """Query the LiteLLM API with user text.
        
        Args:
            user_text: User's query text
            
        Returns:
            Response text from the LLM
        """
        payload = {
            "model": self.model,
            "messages": self._build_messages(user_text),
            "stream": self.stream,
            "max_tokens": self.max_tokens,
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Only add Authorization header if API key is provided
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        print("-> Sending user text to LiteLLM:\n", user_text)

        response = requests.post(
            self.url, 
            json=payload, 
            headers=headers,
            stream=self.stream, 
            timeout=120
        )
        
        if response.status_code != 200:
            print("Error from LiteLLM:", response.status_code, response.text)
            return ""

        response_text = ""
        if self.stream:
            # OpenAI streaming format uses "data: " prefix for each chunk
            for line in response.iter_lines():
                if not line:
                    continue
                
                line_str = line.decode("utf-8")
                
                # Skip empty lines and "data: [DONE]" marker
                if not line_str.startswith("data: ") or line_str == "data: [DONE]":
                    continue
                
                # Remove "data: " prefix
                json_str = line_str[6:]
                
                try:
                    chunk = json.loads(json_str)
                    # Extract content from OpenAI format
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            response_text += content
                except json.JSONDecodeError:
                    continue
        else:
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                response_text = message.get("content", "")

        return response_text.strip()


class LiteLLMClientWithTools:
    """Enhanced LiteLLM client that supports tool calling using OpenAI-compatible API."""
    
    def __init__(
        self,
        url: str,
        model: str,
        api_key: str,
        stream: bool = False,  # Disable streaming for tool support
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize LiteLLM client with tool support.
        
        Args:
            url: Base URL for the LiteLLM API
            model: Model name to use
            api_key: API key for authentication
            stream: Whether to stream responses (should be False for tool support)
            system_prompt: Optional system prompt to prepend to conversations
            max_tokens: Maximum tokens in response (default: 2048)
        """
        self.url = url
        self.model = model
        self.api_key = api_key
        self.stream = stream
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.conversation_history: list[dict] = []
        
        if system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
    
    def query_with_tools(self, user_text: str, tools: list[dict]) -> str:
        """Query LiteLLM with tool support and handle tool calls.
        
        Args:
            user_text: User's query text
            tools: List of tool definitions in OpenAI format
            
        Returns:
            Final response text from the LLM
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_text
        })
        
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Make request to LiteLLM with tools
            payload = {
                "model": self.model,
                "messages": self.conversation_history,
                "stream": self.stream,
                "tools": tools,
                "max_tokens": self.max_tokens,
            }
            
            headers = {
                "Content-Type": "application/json",
            }
            
            # Only add Authorization header if API key is provided
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=120
                )
                
                if response.status_code != 200:
                    print(f"LiteLLM error: {response.status_code}")
                    print(f"Response: {response.text}")
                    return "Sorry, I encountered an error processing your request."
                
                data = response.json()
                
                # Extract message from OpenAI format
                choices = data.get("choices", [])
                if not choices:
                    print(f"Warning: Empty choices in response: {data}")
                    return "Sorry, I received an empty response."
                
                message = choices[0].get("message", {})
                finish_reason = choices[0].get("finish_reason", "")
                
                # Debug: Log finish reason
                if finish_reason:
                    print(f"   Finish reason: {finish_reason}")
                
                # Check if model wants to use tools
                tool_calls = message.get("tool_calls", [])
                
                if tool_calls:
                    print(f"🔧 Model requested {len(tool_calls)} tool call(s)")
                    
                    # Add assistant's tool call message to history
                    self.conversation_history.append(message)
                    
                    # Import execute_tool_call here to avoid circular imports
                    from tools import execute_tool_call
                    
                    # Execute each tool call
                    for tool_call in tool_calls:
                        function = tool_call.get("function", {})
                        tool_name = function.get("name", "")
                        tool_call_id = tool_call.get("id", "")
                        
                        # Parse arguments (might be string or dict)
                        arguments_raw = function.get("arguments", {})
                        if isinstance(arguments_raw, str):
                            try:
                                arguments = json.loads(arguments_raw)
                            except json.JSONDecodeError:
                                arguments = {}
                        else:
                            arguments = arguments_raw
                        
                        print(f"   Calling: {tool_name}({arguments})")
                        print(f"   Tool call ID: {tool_call_id}")
                        
                        # Execute the tool
                        tool_result = execute_tool_call(tool_name, arguments)
                        
                        # Add tool result to conversation history (OpenAI format)
                        tool_response = {
                            "role": "tool",
                            "content": tool_result
                        }
                        
                        # Only add tool_call_id if it exists (some APIs require it, some don't)
                        if tool_call_id:
                            tool_response["tool_call_id"] = tool_call_id
                        
                        self.conversation_history.append(tool_response)
                        print(f"   Tool result length: {len(tool_result)} chars")
                    
                    # Debug: Show conversation state
                    print(f"   Conversation has {len(self.conversation_history)} messages, continuing for final response...")
                    
                    # After tool execution, add a reminder for the model to answer
                    # This helps models that get stuck in tool-calling loops
                    if iteration >= 2:
                        self.conversation_history.append({
                            "role": "user",
                            "content": "Based on the information gathered, please provide a clear and concise answer to my original question. Do not use any more tools."
                        })
                        print(f"   Added reminder to provide final answer (iteration {iteration})")
                    
                    # Continue loop to let model process tool results
                    continue
                
                else:
                    # No tool calls, model provided final answer
                    content = message.get("content", "")
                    
                    # Debug: Check if content is empty
                    if not content or not content.strip():
                        print(f"Warning: Model returned empty content. Message: {message}")
                        print(f"Full response data: {data}")
                        
                        # Try to get content from other possible fields
                        # Some reasoning models put the actual response in 'reasoning_content'
                        if "reasoning_content" in message:
                            reasoning = message.get("reasoning_content", "")
                            if reasoning and reasoning.strip():
                                print(f"   Using reasoning_content as response")
                                content = reasoning
                        elif "text" in message:
                            content = message.get("text", "")
                        elif "delta" in message:
                            content = message.get("delta", {}).get("content", "")
                        
                        # If still empty, return a helpful message
                        if not content or not content.strip():
                            print(f"   No usable content found, returning fallback message")
                            content = "I processed your request but couldn't generate a response. Please try again."
                    
                    # Add assistant's response to history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": content
                    })
                    
                    return content.strip()
                    
            except Exception as e:
                print(f"Error querying LiteLLM: {e}")
                import traceback
                traceback.print_exc()
                return "Sorry, I encountered an error processing your request."
        
        # If we've exhausted iterations, try to give a summary based on conversation
        print(f"⚠️  Reached max iterations ({max_iterations}) without final response")
        print(f"   Conversation history length: {len(self.conversation_history)}")
        
        # Return a fallback message
        return "I gathered information but had trouble formulating a complete response. Please try asking in a different way."


__all__ = ["LiteLLMClient", "LiteLLMClientWithTools"]

