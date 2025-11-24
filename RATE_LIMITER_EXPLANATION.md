# Global Rate Limiter - How It Works

## What It Does

The **Global Rate Limiter** is a thread-safe coordinator that ensures **all agents** (proposer, critic, judge) stay within your API's rate limits, regardless of coordination mode.

### Key Features

1. **Centralized Tracking**: All API requests go through one rate limiter
2. **Thread-Safe**: Multiple agents can call it simultaneously without conflicts
3. **Automatic Waiting**: When at the limit, it automatically waits until a slot is available
4. **Works with All Modes**: Graph, Star, Chain, Tree - all coordination modes work

## How It Works

### The Problem Without Rate Limiter

```
Time: 0:00
Agent 1 (Proposer): Makes request → ✅ Success
Agent 2 (Critic): Makes request → ✅ Success  
Agent 3 (Judge): Makes request → ✅ Success
Agent 1: Makes another request → ✅ Success
Agent 2: Makes another request → ✅ Success
Agent 3: Makes another request → ❌ 429 Error (6th request in < 1 minute)
```

### The Solution With Rate Limiter

```
Time: 0:00
Agent 1 (Proposer): Calls rate_limiter.wait_if_needed() → ✅ Allowed (1/5)
Agent 2 (Critic): Calls rate_limiter.wait_if_needed() → ✅ Allowed (2/5)
Agent 3 (Judge): Calls rate_limiter.wait_if_needed() → ✅ Allowed (3/5)
Agent 1: Calls rate_limiter.wait_if_needed() → ✅ Allowed (4/5)
Agent 2: Calls rate_limiter.wait_if_needed() → ✅ Allowed (5/5)
Agent 3: Calls rate_limiter.wait_if_needed() → ⏳ WAITS 12 seconds (limit reached)
Agent 3: After wait → ✅ Allowed (oldest request expired, now 4/5)
```

## Technical Details

### Request Tracking

The rate limiter maintains a list of request timestamps:

```python
requests = [
    0.0,   # Request at time 0:00
    2.5,   # Request at time 0:02.5
    5.0,   # Request at time 0:05
    10.0,  # Request at time 0:10
    15.0,  # Request at time 0:15
]
```

When a new request comes in:
1. Remove timestamps older than 60 seconds
2. Count remaining requests
3. If < 5, allow immediately
4. If = 5, wait until oldest expires

### Thread Safety

Uses Python's `Lock` to ensure only one agent checks/updates the rate limiter at a time:

```python
with self.lock:  # Only one agent can enter this block at a time
    # Check rate limit
    # Update request list
    # Wait if needed
```

This prevents race conditions where multiple agents might think they can make requests simultaneously.

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Enable/disable rate limiting (default: true)
RATE_LIMIT_ENABLED=true

# Maximum requests per time window (default: 5)
RATE_LIMIT_MAX_REQUESTS=5

# Time window in seconds (default: 60)
RATE_LIMIT_TIME_WINDOW=60
```

### Automatic Detection

The rate limiter automatically enables for Anthropic models:
- `claude-3-haiku-20240307` → Rate limiter ON
- `gpt-3.5-turbo` → Rate limiter OFF (unless enabled manually)
- `gpt-4` → Rate limiter OFF (unless enabled manually)

## Benefits

### ✅ Works with Graph Mode

**Before (without rate limiter):**
- 3 agents in parallel → 3 simultaneous requests
- All hit rate limit → 429 errors
- Exponential backoff delays
- Slow execution

**After (with rate limiter):**
- 3 agents in parallel → Requests are queued
- Rate limiter coordinates → No 429 errors
- Smooth execution
- All coordination modes work

### ✅ No Code Changes Needed

Once integrated, it works automatically:
- No changes to agent code
- No changes to engine code
- Works with existing retry logic
- Transparent to agents

### ✅ Flexible Configuration

- Adjust limits per API
- Enable/disable as needed
- Works with any coordination mode

## Example: Graph Mode with Rate Limiter

```
Iteration 1:
├─ Proposer: plan_task() → rate_limiter.wait() → ✅ (1/5)
├─ Critic: plan_task() → rate_limiter.wait() → ✅ (2/5)
├─ Judge: plan_task() → rate_limiter.wait() → ✅ (3/5)
├─ Proposer: act() → rate_limiter.wait() → ✅ (4/5)
├─ Critic: act() → rate_limiter.wait() → ✅ (5/5)
└─ Judge: act() → rate_limiter.wait() → ⏳ Wait 12s → ✅ (1/5, old expired)

Iteration 2:
├─ Proposer: plan_task() → rate_limiter.wait() → ✅ (2/5)
└─ ... continues smoothly
```

## Comparison: With vs Without

### Without Rate Limiter (Graph Mode)
- ❌ Multiple 429 errors
- ❌ Exponential backoff delays (1s, 2s, 4s, 8s...)
- ❌ Unpredictable execution time
- ❌ Agents may get stuck

### With Rate Limiter (Graph Mode)
- ✅ No 429 errors
- ✅ Predictable delays (only when needed)
- ✅ Smooth execution
- ✅ All agents complete successfully

## Testing All Coordination Modes

With the rate limiter, you can now test all modes:

```yaml
# Test 1: Graph mode
coordinate_mode: graph  # ✅ Works - rate limiter coordinates parallel requests

# Test 2: Star mode  
coordinate_mode: star   # ✅ Works - rate limiter coordinates sequential requests

# Test 3: Chain mode
coordinate_mode: chain  # ✅ Works - rate limiter coordinates chain requests

# Test 4: Tree mode
coordinate_mode: tree   # ✅ Works - rate limiter coordinates hierarchical requests
```

All modes will work smoothly without rate limit errors!

## Monitoring

The rate limiter logs its activity:

```
INFO:RateLimiter: Rate limit reached (5/5). Waiting 12.3 seconds...
INFO:RateLimiter: Request allowed. Current rate: 3/5 requests in last 60s
```

You can check the status programmatically:
```python
from marble.utils.rate_limiter import get_rate_limiter

limiter = get_rate_limiter()
status = limiter.get_status()
print(f"Available slots: {status['available_slots']}")
```

## Summary

The global rate limiter:
- ✅ Coordinates all API requests across all agents
- ✅ Prevents rate limit errors (429)
- ✅ Works with all coordination modes
- ✅ Thread-safe and automatic
- ✅ Configurable via environment variables
- ✅ Enables testing all coordination modes without rate limit issues

**Result**: You can use graph mode (or any mode) with parallel agents and never hit rate limits!

