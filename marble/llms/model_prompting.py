import os

import litellm
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional
from litellm.types.utils import Message

from marble.llms.error_handler import api_calling_error_exponential_backoff
from marble.utils.rate_limiter import get_rate_limiter


@beartype
@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def model_prompting(
    llm_model: str,
    messages: List[Dict[str, str]],
    return_num: Optional[int] = 1,
    max_token_num: Optional[int] = 512,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    mode: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> List[Message]:
    """
    Select model via router in LiteLLM with support for function calling.
    
    Includes global rate limiting to coordinate requests across all agents.
    """
    # Global rate limiting - coordinates requests across all agents
    # Only enable if RATE_LIMIT_ENABLED env var is set or if using Anthropic
    rate_limit_enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    is_anthropic = "claude" in llm_model.lower() or "anthropic" in llm_model.lower()
    
    if rate_limit_enabled or is_anthropic:
        # Get rate limit settings from environment or use defaults
        # Handle comments in env vars (strip everything after #)
        max_requests_str = os.getenv("RATE_LIMIT_MAX_REQUESTS", "5")
        if max_requests_str:
            # Remove comments (everything after #) and strip whitespace
            max_requests_str = max_requests_str.split("#")[0].strip()
        max_requests = int(max_requests_str) if max_requests_str else 5
        
        time_window_str = os.getenv("RATE_LIMIT_TIME_WINDOW", "60")
        if time_window_str:
            # Remove comments (everything after #) and strip whitespace
            time_window_str = time_window_str.split("#")[0].strip()
        time_window = int(time_window_str) if time_window_str else 60
        
        rate_limiter = get_rate_limiter(max_requests=max_requests, time_window=time_window)
        rate_limiter.wait_if_needed()
    
    # litellm.set_verbose=True
    if "together_ai/TA" in llm_model:
        base_url = "https://api.ohmygpt.com/v1"
    else:
        base_url = None
    
    # Claude models don't allow both temperature and top_p to be specified
    # If both are provided, prefer temperature and set top_p to None
    is_anthropic = "claude" in llm_model.lower() or "anthropic" in llm_model.lower()
    if is_anthropic and temperature is not None and top_p is not None:
        top_p = None  # Claude models prefer temperature over top_p
    
    completion = litellm.completion(
        model=llm_model,
        messages=messages,
        max_tokens=max_token_num,
        n=return_num,
        top_p=top_p,
        temperature=temperature,
        stream=stream,
        tools=tools,
        tool_choice=tool_choice,
        base_url=base_url,
    )
    message_0: Message = completion.choices[0].message
    assert message_0 is not None
    assert isinstance(message_0, Message)
    return [message_0]
