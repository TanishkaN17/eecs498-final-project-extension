# Coordination Modes - No Need to Redefine Agents!

## ✅ Short Answer: **NO, you don't need to redefine agents!**

The **agents** stay the same. Only the **coordination mode** changes how they execute.

---

## 🔄 How Coordination Modes Work

### Your Current Setup:

```yaml
agents:
  - agent_id: proposer
  - agent_id: critic
  - agent_id: judge

relationships:
  - [proposer, critic, "debates_with"]
  - [proposer, judge, "presents_to"]
  - [critic, judge, "presents_to"]
```

**The agents and relationships stay the same!** Only the `coordinate_mode` changes execution.

---

## 📊 Coordination Mode Options

### 1. **Chain Mode** (Current - Fastest for Claude)

```yaml
coordinate_mode: chain
```

**How it works:**

- Agents execute **one at a time** in sequence
- Proposer → Critic → Judge → Proposer → ...
- **Best for Claude**: Reduces parallel requests = fewer rate limit waits

**Execution flow:**

```
Iteration 1: Proposer acts
Iteration 2: Critic acts
Iteration 3: Judge acts
Iteration 4: Proposer acts again
```

### 2. **Graph Mode** (Parallel - Can be slow with Claude)

```yaml
coordinate_mode: graph
```

**How it works:**

- Agents execute **in parallel** based on relationships
- All agents can act simultaneously
- **Problem with Claude**: 3 agents × parallel = hits 5 req/min limit fast

**Execution flow:**

```
Iteration 1: Proposer + Critic + Judge all act at once
Iteration 2: All act again (if needed)
```

### 3. **Star Mode** (Centralized)

```yaml
coordinate_mode: star
```

**How it works:**

- One central agent coordinates others
- Typically the judge would be the center
- Others communicate through the center

**Execution flow:**

```
Judge coordinates → Proposer acts → Judge coordinates → Critic acts → Judge decides
```

### 4. **Tree Mode** (Hierarchical)

```yaml
coordinate_mode: tree
```

**How it works:**

- Hierarchical structure
- Parent-child relationships
- Judge at top, proposer/critic below

**Execution flow:**

```
Judge (root)
  ├─ Proposer (child)
  └─ Critic (child)
```

---

## 🎯 For Your Translation Task

### Recommended: **Chain Mode** (Current)

- ✅ **Best for Claude** (5 req/min limit)
- ✅ **No parallel requests** = no rate limit conflicts
- ✅ **Agents still follow relationships** (proposer debates with critic, both present to judge)
- ✅ **Same agents, same relationships** - just sequential execution

### If You Want to Test Others:

**Star Mode:**

```yaml
coordinate_mode: star
# Judge becomes the central coordinator
# Agents stay the same!
```

**Graph Mode:**

```yaml
coordinate_mode: graph
# All agents act in parallel
# Will be slower with Claude due to rate limits
# But rate limiter will handle it (with waits)
```

**Tree Mode:**

```yaml
coordinate_mode: tree
# Hierarchical structure
# Judge at top, proposer/critic below
```

---

## 🔑 Key Points

1. **Agents don't change** - same `agent_id`, same `profile`
2. **Relationships don't change** - same `relationships` section
3. **Only `coordinate_mode` changes** - just the execution pattern
4. **Rate limiter works with all modes** - but chain is fastest for Claude

---

## 📝 Example: Switching Modes

**To switch from chain to graph:**

```yaml
# Just change this one line:
coordinate_mode: graph # Changed from "chain"

# Everything else stays the same:
agents: # ← Same agents
  - agent_id: proposer
  - agent_id: critic
  - agent_id: judge

relationships: # ← Same relationships
  - [proposer, critic, "debates_with"]
  - [proposer, judge, "presents_to"]
  - [critic, judge, "presents_to"]
```

**That's it!** No need to redefine anything.

---

## ✅ Summary

- **Agents**: Defined once, work with all coordination modes
- **Relationships**: Define how agents interact (same for all modes)
- **Coordination Mode**: Just changes execution pattern
- **For Claude**: Use `chain` mode to avoid rate limit delays
- **To test others**: Just change `coordinate_mode` - nothing else needed!

Your current setup with `chain` mode is perfect for Claude! 🎉
