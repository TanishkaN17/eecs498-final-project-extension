# MARBLE Framework - Translation Task Guide

## Overview

**MARBLE** (Multi-Agent Coordination Backbone with LLM Engine) is a framework for coordinating multiple AI agents to work together on complex tasks. This guide explains how to use it for translation tasks.

## Architecture Overview

### Key Components

1. **Engine** (`marble/engine/engine.py`)

   - Orchestrates the entire simulation
   - Manages agent coordination modes: `star`, `graph`, `chain`, or `tree`
   - Coordinates task distribution and result collection

2. **Agents** (`marble/agent/base_agent.py`)

   - Individual AI agents with profiles/personas
   - Can communicate with each other
   - Execute tasks using LLM models
   - Have memory to remember past actions

3. **Environments** (`marble/environments/`)

   - Define the context where agents operate
   - Provide action handlers (functions agents can call)
   - Track state and task completion

4. **Config** (`marble/configs/config.py`)
   - YAML configuration files define:
     - Agents and their profiles
     - Environment type
     - Task description
     - Coordination mode
     - LLM models to use

## How It Works

### Execution Flow

1. **Load Configuration**: Reads a YAML config file
2. **Initialize Components**: Creates environment, agents, memory, evaluator
3. **Start Coordination**: Based on mode (star/graph/chain/tree):
   - **Star**: Central planner assigns tasks to agents
   - **Graph**: Agents work in parallel, can communicate
   - **Chain**: Agents pass tasks sequentially
   - **Tree**: Hierarchical task delegation
4. **Agent Actions**: Each agent:
   - Receives a task
   - Plans what to do
   - Calls environment functions or communicates with other agents
   - Returns results
5. **Iteration**: Process repeats until task is complete or max iterations reached
6. **Output**: Results written to JSONL file

## Using MARBLE for Translation Tasks

### Step 1: Create a Translation Config File

Create a file `translation_config.yaml`:

```yaml
coordinate_mode: graph # or "star", "chain", "tree"
llm: "gpt-4" # or "gpt-3.5-turbo", "claude-3-opus", etc.

environment:
  type: Base
  name: "Translation Environment"
  max_iterations: 5
  description: "An environment for translation tasks"

task:
  content: "Translate the following text from English to Spanish: 'Hello, how are you today? I hope you are having a wonderful day.'"
  output_format: "Provide the translation in Spanish."

agents:
  - type: BaseAgent
    agent_id: translator
    profile: "You are an expert translator specializing in English to Spanish translation. You provide accurate, natural translations that preserve the meaning and tone of the original text."

  - type: BaseAgent
    agent_id: reviewer
    profile: "You are a translation quality reviewer. You check translations for accuracy, fluency, and cultural appropriateness."

relationships:
  - [translator, reviewer, "collaborates_with"]

memory:
  type: SharedMemory

engine_planner:
  planning_method: "naive"
  initial_progress: "Starting translation task."

output:
  file_path: "translation_output.jsonl"

metrics:
  # Evaluation metrics configuration
```

### Step 2: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
# Or for other providers:
# TOGETHER_API_KEY=your_together_api_key
# ANTHROPIC_API_KEY=your_anthropic_api_key
```

### Step 3: Install Dependencies

```bash
cd MARBLE
poetry install
# Or if using pip:
# pip install -r requirements.txt
```

### Step 4: Run the Translation Task

```bash
cd MARBLE
python -m marble.main --config_path marble/configs/translation_config.yaml
```

## Creating a Custom Translation Environment (Optional)

For more control, you can create a custom environment with translation-specific actions:

1. Create `marble/environments/translation_env.py`:

```python
from marble.environments.base_env import BaseEnvironment

class TranslationEnvironment(BaseEnvironment):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self._register_translation_actions()

    def _register_translation_actions(self):
        # Register a translate action
        self.register_action(
            action_name="translate_text",
            handler=self._translate_text,
            description={
                "type": "function",
                "function": {
                    "name": "translate_text",
                    "description": "Translate text from source language to target language",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to translate"},
                            "source_lang": {"type": "string", "description": "Source language code"},
                            "target_lang": {"type": "string", "description": "Target language code"}
                        },
                        "required": ["text", "source_lang", "target_lang"]
                    }
                }
            }
        )

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> dict:
        # This would use an LLM or translation API
        # For now, return a placeholder
        return {
            "translated_text": f"[Translation of '{text}' from {source_lang} to {target_lang}]",
            "source_lang": source_lang,
            "target_lang": target_lang
        }
```

2. Register it in `marble/environments/__init__.py`
3. Update `marble/engine/engine.py` to handle "Translation" environment type

## Coordination Modes Explained

### Graph Mode (Recommended for Translation)

- Agents work in parallel
- Can communicate with each other
- Good for: Multi-step translation with review/editing

### Star Mode

- Central planner assigns tasks
- Agents don't directly communicate
- Good for: Simple translation with quality check

### Chain Mode

- Sequential task passing
- One agent completes, passes to next
- Good for: Translation pipeline (translate → review → edit)

### Tree Mode

- Hierarchical delegation
- Parent agents assign to children
- Good for: Complex translation with multiple reviewers

## Example Multi-Agent Translation Workflow

For a sophisticated translation task, you might use:

1. **Translator Agent**: Performs initial translation
2. **Reviewer Agent**: Checks accuracy and fluency
3. **Editor Agent**: Refines the translation
4. **Finalizer Agent**: Ensures consistency and formatting

Each agent can communicate with others to improve the final output.

## Output Format

Results are saved to a JSONL file with:

- Task description
- Iteration-by-iteration progress
- Agent communications
- Final translation
- Evaluation scores
- Token usage statistics

## Tips for Translation Tasks

1. **Use appropriate agent profiles**: Make them translation experts
2. **Set clear task descriptions**: Specify source/target languages
3. **Enable communication**: Use `graph` mode for agent collaboration
4. **Adjust max_iterations**: More complex translations may need more iterations
5. **Use quality reviewers**: Add agents to review and improve translations

## Troubleshooting

- **API Key Issues**: Ensure `.env` file is properly configured
- **Import Errors**: Run `poetry install` or `pip install -r requirements.txt`
- **Config Errors**: Validate YAML syntax
- **Memory Issues**: Reduce max_iterations or number of agents
