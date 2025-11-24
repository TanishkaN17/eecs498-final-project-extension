# Rate Limit Problem & Solutions

## The Problem

Your Anthropic API has a **rate limit of 5 requests per minute**. With 3 agents (proposer, critic, judge) working in parallel in graph mode, they're all making API calls simultaneously, causing:

- Multiple `429 Too Many Requests` errors
- Exponential backoff delays (1s, 2s, 4s, 8s...)
- Agents getting stuck waiting for retries
- Slow execution or failures

## Current Status

From your logs:
- ✅ Agents are working correctly
- ✅ They successfully called `get_current_input()` 
- ✅ They got the Hindi text: "नमस्ते, आप कैसे हैं?"
- ❌ They're hitting rate limits before calling `submit_translation()`
- ❌ Judge is waiting but can't proceed

## Solutions

### Solution 1: Add Delays Between Agent Actions (Quick Fix)

Modify the engine to add delays between agent executions in graph mode:

```python
# In marble/engine/engine.py, graph_coordinate() method
import time

# After line ~341, add delay:
for agent in current_agents:
    time.sleep(12)  # Wait 12 seconds between agents (5 req/min = 12 sec/req)
    # ... rest of agent execution
```

### Solution 2: Use Sequential Execution (Recommended)

Change coordination mode to `chain` or `star` instead of `graph`:

**Option A: Chain Mode** (Sequential)
```yaml
coordinate_mode: chain
```
- Proposer → Critic → Judge (one at a time)
- No parallel requests
- Slower but avoids rate limits

**Option B: Star Mode** (Centralized)
```yaml
coordinate_mode: star
```
- Central planner assigns tasks sequentially
- Agents execute one at a time
- Better control over timing

### Solution 3: Switch to a Model with Higher Rate Limits

**Use OpenAI instead:**
```yaml
llm: "gpt-3.5-turbo"  # or "gpt-4"
```
- OpenAI has much higher rate limits (typically 60+ requests/minute)
- Update your `.env` file with `OPENAI_API_KEY`

**Or use a different Anthropic model:**
- Claude Sonnet/Opus may have different limits
- Check your Anthropic dashboard for limits

### Solution 4: Add Global Rate Limiter (Advanced)

Create a rate limiter that coordinates all agent requests:

```python
# Create marble/utils/rate_limiter.py
import time
from threading import Lock

class RateLimiter:
    def __init__(self, max_requests: int = 5, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove old requests outside time window
            self.requests = [r for r in self.requests if now - r < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                # Wait until oldest request expires
                wait_time = self.time_window - (now - self.requests[0]) + 1
                time.sleep(wait_time)
                self.requests = [r for r in self.requests if now - r < self.time_window]
            
            self.requests.append(now)
```

Then use it in `model_prompting.py`:
```python
from marble.utils.rate_limiter import RateLimiter

rate_limiter = RateLimiter(max_requests=5, time_window=60)

@beartype
@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def model_prompting(...):
    rate_limiter.wait_if_needed()  # Add this line
    completion = litellm.completion(...)
```

## Recommended Quick Fix

**For immediate use, change your config to chain mode:**

```yaml
coordinate_mode: chain  # Instead of "graph"
```

This will make agents execute sequentially, avoiding rate limit conflicts.

## Long-term Solution

1. **Switch to OpenAI** (if possible) - much higher rate limits
2. **Or implement Solution 4** - global rate limiter for better control
3. **Or upgrade Anthropic plan** - contact sales for higher limits

## Why This Happens

- **Graph mode** = All agents work in parallel
- **3 agents** × **multiple API calls each** = 10+ requests quickly
- **5 requests/minute limit** = Can't handle parallel execution
- **Result** = Rate limit errors and delays

The retry logic helps, but when all agents retry at once, they still compete for the same limited slots.

