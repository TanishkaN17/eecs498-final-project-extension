# Translation Environment Setup Guide

## Overview

The Translation Environment provides a complete framework for multi-agent Hindi-to-English translation with automatic evaluation using COMET and BLEURT metrics.

## Features

✅ **File Input**: Load Hindi text from `.txt` file (one line per input)  
✅ **Automatic Evaluation**: COMET and BLEURT scores for each translation  
✅ **Iterative Rounds**: Judge can request multiple debate rounds  
✅ **Structured Actions**: Agents use environment functions for translation workflow  
✅ **Evaluation Tracking**: All translations are evaluated and scored

## Installation

### 1. Install Base Dependencies

```bash
cd MARBLE
poetry install
# Or: pip install -r requirements.txt
```

### 2. Install Translation Evaluation Dependencies

```bash
pip install -r translation_requirements.txt
```

This installs:

- `unbabel-comet` - COMET evaluation metric
- `bleurt` - BLEURT evaluation metric
- `tensorflow` - Required for BLEURT

### 3. Download Evaluation Models

**COMET Model** (automatic):

- The environment will automatically download the COMET model on first use
- Model: `Unbabel/wmt22-comet-da`
- Location: `~/.comet/models/unbabel-comet/wmt22-comet-da`

**BLEURT Model** (manual):

1. Download BLEURT-20 from: https://github.com/google-research/bleurt
2. Extract to: `~/.bleurt/BLEURT-20`
3. Or use the command:
   ```bash
   mkdir -p ~/.bleurt
   cd ~/.bleurt
   wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
   unzip BLEURT-20.zip
   ```

**Note**: If BLEURT model is not available, the environment will still work but BLEURT scores will be `None`.

## Configuration

### Input File Format

Create a file `input_hindi.txt` with one Hindi sentence per line:

```
नमस्ते, आप कैसे हैं?
मुझे आशा है कि आपका दिन अच्छा बीत रहा है।
आज मौसम बहुत सुंदर है।
```

### Config File Settings

In `marble/configs/translation_config.yaml`:

```yaml
environment:
  type: Translation
  input_file: "input_hindi.txt" # Path to your Hindi input file
  source_lang: "hi" # Hindi
  target_lang: "en" # English
  enable_evaluation: true # Enable COMET and BLEURT
```

## How It Works

### 1. **Input Loading**

- Environment loads Hindi texts from `input_hindi.txt`
- Each line becomes an input with ID: `input_0`, `input_1`, etc.
- Agents process one input at a time

### 2. **Translation Submission**

- **Proposer** calls `get_current_input()` → gets Hindi text
- **Proposer** calls `submit_translation(translation, rationale)` → submits English translation
- Translation is **automatically evaluated** with COMET (and BLEURT if reference available)

### 3. **Debate Process**

- **Critic** calls `get_other_translations()` → sees proposer's translation and scores
- **Critic** calls `submit_translation()` → submits alternative translation
- Both agents communicate and debate
- Both present to **Judge**

### 4. **Judge Decision**

- **Judge** calls `get_other_translations()` → sees both translations and scores
- **Judge** calls `judge_decision()`:
  - `decision="finalize"` → Accepts a translation, moves to next input
  - `decision="another_round"` → Requests new translations, increments round counter

### 5. **Iterative Rounds**

- If judge requests another round, `current_round` increments
- Proposer and Critic submit new translations
- Process repeats until judge finalizes

## Available Actions

### For All Agents

1. **`get_current_input()`**

   - Returns current Hindi text to translate
   - Returns: `{input_id, text, source_lang, target_lang}`

2. **`get_all_inputs()`**
   - Returns all input texts
   - Useful for batch processing

### For Proposer/Critic

3. **`submit_translation(translation, rationale, input_id?)`**

   - Submit your translation proposal
   - Automatically evaluated with COMET (and BLEURT if reference available)
   - Returns: `{success, evaluation: {comet_score, bleurt_score}}`

4. **`get_other_translations(input_id?, round?)`**
   - Get translations from other agents
   - See their evaluation scores
   - Returns: `{other_translations: [{agent_id, translation, rationale, evaluation}]}`

### For Judge

5. **`judge_decision(decision, reasoning, final_translation?, input_id?)`**
   - Make final decision
   - `decision="finalize"` → Accept translation, move to next input
   - `decision="another_round"` → Request more debate
   - Returns: `{success, decision, evaluation}` (if finalizing)

## Evaluation Metrics

### COMET (Crosslingual Optimized Metric for Evaluation of Translation)

- **Range**: Typically -1 to 1 (higher is better)
- **Type**: Quality estimation (works without reference)
- **Model**: `Unbabel/wmt22-comet-da`
- **Automatic**: Evaluates every `submit_translation()` call

### BLEURT (Bilingual Evaluation Understudy with Representations from Transformers)

- **Range**: Typically -1 to 1 (higher is better)
- **Type**: Requires reference translation
- **Model**: BLEURT-20
- **Note**: Only available if reference translations are provided

## Output

Results are saved to `translation_output.jsonl` with:

- All submitted translations
- Evaluation scores (COMET, BLEURT)
- Final translations
- Round-by-round history
- Judge decisions

## Example Workflow

```
Round 1:
1. Proposer: get_current_input() → "नमस्ते, आप कैसे हैं?"
2. Proposer: submit_translation("Hello, how are you?", rationale="...")
   → COMET: 0.85
3. Critic: get_other_translations() → sees proposer's translation
4. Critic: submit_translation("Hi, how are you doing?", rationale="...")
   → COMET: 0.82
5. Judge: get_other_translations() → sees both
6. Judge: judge_decision(decision="another_round", reasoning="Need better fluency")

Round 2:
1. Proposer: submit_translation("Hello, how are you today?", rationale="...")
   → COMET: 0.91
2. Critic: submit_translation("Hi there, how are you doing?", rationale="...")
   → COMET: 0.89
3. Judge: judge_decision(decision="finalize", final_translation="Hello, how are you today?")
   → Final COMET: 0.91
   → Moves to next input
```

## Troubleshooting

### COMET Model Not Loading

- Check internet connection (first download requires internet)
- Model will be cached in `~/.comet/models/`
- If issues persist, manually download and place in cache directory

### BLEURT Not Working

- Ensure BLEURT-20 model is in `~/.bleurt/BLEURT-20`
- Check TensorFlow installation
- BLEURT is optional - environment works without it

### No Evaluation Scores

- Check `enable_evaluation: true` in config
- Ensure models are loaded (check logs)
- Scores will be `None` if models unavailable (but system still works)

### Input File Not Found

- Ensure `input_file` path in config is correct
- Use relative path from MARBLE directory or absolute path
- Check file encoding (should be UTF-8)

## Running the Translation Task

```bash
cd MARBLE
python -m marble.main --config_path marble/configs/translation_config.yaml
```

Or use the provided script:

```bash
bash run_translation.sh
# Or on Windows:
run_translation.ps1
```

## Next Steps

1. Create your `input_hindi.txt` file with Hindi text
2. Update config file with correct paths
3. Set your API keys in `.env` file
4. Run the translation task
5. Check `translation_output.jsonl` for results
