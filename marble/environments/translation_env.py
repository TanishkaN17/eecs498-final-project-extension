"""
Translation environment module for multi-agent translation tasks.
Supports Hindi to English translation with COMET and BLEURT evaluation.
"""

import os
import time
from typing import Any, Dict, List, Optional

from marble.environments.base_env import BaseEnvironment
from marble.utils.logger import get_logger

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Warning: COMET not available. Install with: pip install unbabel-comet")

BLEURT_AVAILABLE = False
try:
    import tensorflow as tf
    # Try to import bleurt - the import structure may vary
    try:
        from bleurt import score
        BLEURT_AVAILABLE = True
    except ImportError:
        try:
            import bleurt
            BLEURT_AVAILABLE = True
        except ImportError:
            pass
except ImportError:
    pass

if not BLEURT_AVAILABLE:
    print("Warning: BLEURT not available. Install with: pip install bleurt")


class TranslationEnvironment(BaseEnvironment):
    """
    Environment for translation tasks with evaluation metrics.
    Supports iterative debate rounds and automatic evaluation.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the TranslationEnvironment.

        Args:
            name (str): Name of the environment
            config (Dict[str, Any]): Configuration dictionary
        """
        super().__init__(name, config)
        self.logger = get_logger(self.__class__.__name__)
        
        # Translation state
        self.source_lang = config.get("source_lang", "hi")  # Hindi
        self.target_lang = config.get("target_lang", "en")  # English
        self.translations: Dict[str, List[Dict[str, Any]]] = {}  # {agent_id: [translations]}
        self.translation_history: List[Dict[str, Any]] = []
        self.current_round = 0
        self.final_translations: Dict[str, str] = {}  # {input_line_id: final_translation}
        
        # Input file handling
        self.input_file = config.get("input_file", None)
        self.input_texts: List[Dict[str, str]] = []  # [{id, text}]
        self.current_input_index = 0
        
        # Evaluation models
        self.comet_model = None
        self.bleurt_scorer = None
        self.evaluation_enabled = config.get("enable_evaluation", True)
        
        # Track current agent for evaluation
        self.current_agent_id: Optional[str] = None
        
        # Load input texts
        if self.input_file:
            self._load_input_file()
        
        # Initialize evaluation models
        if self.evaluation_enabled:
            self._initialize_evaluation_models()
        
        # Register translation actions
        self._register_translation_actions()
        
        self.logger.info(f"TranslationEnvironment initialized: {len(self.input_texts)} inputs loaded")

    def _load_input_file(self) -> None:
        """Load input texts from file (one per line)."""
        if not os.path.exists(self.input_file):
            self.logger.warning(f"Input file not found: {self.input_file}")
            return
        
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            self.input_texts = [
                {"id": f"input_{i}", "text": line.strip()}
                for i, line in enumerate(lines)
                if line.strip()  # Skip empty lines
            ]
            
            self.logger.info(f"Loaded {len(self.input_texts)} input texts from {self.input_file}")
        except Exception as e:
            self.logger.error(f"Error loading input file: {e}")
            self.input_texts = []

    def _initialize_evaluation_models(self) -> None:
        """Initialize COMET and BLEURT evaluation models."""
        # Initialize COMET
        if COMET_AVAILABLE:
            try:
                comet_model_path = os.path.expanduser("~/.comet/models/unbabel-comet/wmt22-comet-da")
                if not os.path.exists(comet_model_path):
                    self.logger.info("Downloading COMET model...")
                    model_path = download_model("Unbabel/wmt22-comet-da")
                else:
                    model_path = comet_model_path
                
                self.comet_model = load_from_checkpoint(model_path)
                self.logger.info("COMET model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load COMET model: {e}")
                self.comet_model = None
        
        # Initialize BLEURT
        if BLEURT_AVAILABLE:
            try:
                # Using BLEURT-20 model
                bleurt_model_path = os.path.expanduser("~/.bleurt/BLEURT-20")
                if not os.path.exists(bleurt_model_path):
                    self.logger.warning("BLEURT model not found. Please download BLEURT-20 manually.")
                    self.bleurt_scorer = None
                else:
                    # Try different import patterns
                    try:
                        from bleurt import score
                        self.bleurt_scorer = score.BleurtScorer(bleurt_model_path)
                    except:
                        try:
                            import bleurt
                            self.bleurt_scorer = bleurt.score.BleurtScorer(bleurt_model_path)
                        except:
                            self.logger.warning("Could not initialize BLEURT scorer")
                            self.bleurt_scorer = None
                    if self.bleurt_scorer:
                        self.logger.info("BLEURT model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load BLEURT model: {e}")
                self.bleurt_scorer = None

    def _evaluate_translation(
        self, 
        source: str, 
        translation: str, 
        reference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate translation using COMET and BLEURT.

        Args:
            source: Source text (Hindi)
            translation: Translated text (English)
            reference: Reference translation (optional)

        Returns:
            Dict with evaluation scores
        """
        scores = {
            "comet_score": None,
            "bleurt_score": None,
            "evaluation_time": time.time()
        }
        
        # COMET evaluation
        if self.comet_model and COMET_AVAILABLE:
            try:
                if reference:
                    # With reference
                    data = [{"src": source, "mt": translation, "ref": reference}]
                    scores_data = self.comet_model.predict(data, batch_size=8, gpus=0)
                    scores["comet_score"] = float(scores_data.scores[0])
                else:
                    # Without reference (quality estimation)
                    data = [{"src": source, "mt": translation}]
                    scores_data = self.comet_model.predict(data, batch_size=8, gpus=0)
                    scores["comet_score"] = float(scores_data.scores[0])
            except Exception as e:
                self.logger.warning(f"COMET evaluation failed: {e}")
        
        # BLEURT evaluation (requires reference)
        if self.bleurt_scorer and BLEURT_AVAILABLE and reference:
            try:
                bleurt_scores = self.bleurt_scorer.score(
                    references=[reference],
                    candidates=[translation]
                )
                scores["bleurt_score"] = float(bleurt_scores[0])
            except Exception as e:
                self.logger.warning(f"BLEURT evaluation failed: {e}")
        
        scores["evaluation_time"] = time.time() - scores["evaluation_time"]
        return scores

    def _register_translation_actions(self) -> None:
        """Register all translation-related actions."""
        
        # Action 1: Submit translation
        submit_translation_desc = {
            "type": "function",
            "function": {
                "name": "submit_translation",
                "description": "Submit your translation proposal with rationale. This will be evaluated automatically.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "translation": {
                            "type": "string",
                            "description": "Your English translation of the Hindi text"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why this translation is good"
                        },
                        "input_id": {
                            "type": "string",
                            "description": "ID of the input text being translated (optional, will use current if not provided)"
                        }
                    },
                    "required": ["translation", "rationale"]
                }
            }
        }
        self.register_action(
            "submit_translation",
            self._submit_translation,
            submit_translation_desc
        )
        
        # Action 2: Get other translations
        get_translations_desc = {
            "type": "function",
            "function": {
                "name": "get_other_translations",
                "description": "Get translations submitted by other agents for comparison",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_id": {
                            "type": "string",
                            "description": "ID of the input text (optional)"
                        },
                        "round": {
                            "type": "integer",
                            "description": "Round number (optional, defaults to current round)"
                        }
                    }
                }
            }
        }
        self.register_action(
            "get_other_translations",
            self._get_other_translations,
            get_translations_desc
        )
        
        # Action 3: Get current input text
        get_input_desc = {
            "type": "function",
            "function": {
                "name": "get_current_input",
                "description": "Get the current Hindi text that needs to be translated",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
        self.register_action(
            "get_current_input",
            self._get_current_input,
            get_input_desc
        )
        
        # Action 4: Judge decision (finalize or request another round)
        judge_decision_desc = {
            "type": "function",
            "function": {
                "name": "judge_decision",
                "description": "Make a final decision: either finalize a translation or request another round of debate",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": ["finalize", "another_round"],
                            "description": "Either 'finalize' to accept a translation, or 'another_round' to request more debate"
                        },
                        "final_translation": {
                            "type": "string",
                            "description": "The final chosen translation (required if decision is 'finalize')"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of your decision"
                        },
                        "input_id": {
                            "type": "string",
                            "description": "ID of the input text (optional)"
                        }
                    },
                    "required": ["decision", "reasoning"]
                }
            }
        }
        self.register_action(
            "judge_decision",
            self._judge_decision,
            judge_decision_desc
        )
        
        # Action 5: Get all inputs (for batch processing)
        get_all_inputs_desc = {
            "type": "function",
            "function": {
                "name": "get_all_inputs",
                "description": "Get all input texts that need to be translated",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
        self.register_action(
            "get_all_inputs",
            self._get_all_inputs,
            get_all_inputs_desc
        )

    def _submit_translation(
        self, 
        translation: str, 
        rationale: str, 
        input_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit a translation proposal.

        Args:
            translation: The English translation
            rationale: Explanation for the translation
            input_id: ID of the input text (optional)

        Returns:
            Dict with submission result and evaluation scores
        """
        # Get current input if not specified
        if not input_id:
            if self.current_input_index < len(self.input_texts):
                input_id = self.input_texts[self.current_input_index]["id"]
                source_text = self.input_texts[self.current_input_index]["text"]
            else:
                return {
                    "success": False,
                    "error": "No input text available"
                }
        else:
            # Find the input text
            input_item = next((item for item in self.input_texts if item["id"] == input_id), None)
            if not input_item:
                return {
                    "success": False,
                    "error": f"Input ID {input_id} not found"
                }
            source_text = input_item["text"]
        
        # Store translation
        agent_id = self.current_agent_id or "unknown"
        
        if agent_id not in self.translations:
            self.translations[agent_id] = []
        
        translation_entry = {
            "translation": translation,
            "rationale": rationale,
            "input_id": input_id,
            "source_text": source_text,
            "round": self.current_round,
            "timestamp": time.time(),
            "evaluation": {}
        }
        
        # Evaluate translation
        if self.evaluation_enabled:
            eval_scores = self._evaluate_translation(
                source=source_text,
                translation=translation
            )
            translation_entry["evaluation"] = eval_scores
        
        self.translations[agent_id].append(translation_entry)
        self.translation_history.append({
            "agent_id": agent_id,
            **translation_entry
        })
        
        self.logger.info(
            f"Translation submitted by {agent_id} for {input_id} "
            f"(Round {self.current_round}): COMET={translation_entry['evaluation'].get('comet_score', 'N/A')}, "
            f"BLEURT={translation_entry['evaluation'].get('bleurt_score', 'N/A')}"
        )
        
        return {
            "success": True,
            "message": f"Translation submitted successfully (Round {self.current_round})",
            "input_id": input_id,
            "round": self.current_round,
            "evaluation": translation_entry["evaluation"]
        }

    def _get_other_translations(
        self, 
        input_id: Optional[str] = None,
        round: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get translations from other agents."""
        if round is None:
            round = self.current_round
        
        if not input_id:
            if self.current_input_index < len(self.input_texts):
                input_id = self.input_texts[self.current_input_index]["id"]
            else:
                return {
                    "success": False,
                    "error": "No input text available"
                }
        
        current_agent = self.current_agent_id or "unknown"
        other_translations = []
        
        for agent_id, translations in self.translations.items():
            if agent_id == current_agent:
                continue
            
            # Get translations for this input and round
            for trans in translations:
                if trans["input_id"] == input_id and trans["round"] == round:
                    other_translations.append({
                        "agent_id": agent_id,
                        "translation": trans["translation"],
                        "rationale": trans["rationale"],
                        "evaluation": trans.get("evaluation", {})
                    })
        
        return {
            "success": True,
            "other_translations": other_translations,
            "count": len(other_translations),
            "input_id": input_id,
            "round": round
        }

    def _get_current_input(self) -> Dict[str, Any]:
        """Get the current input text to translate."""
        if self.current_input_index >= len(self.input_texts):
            return {
                "success": False,
                "error": "No more input texts available"
            }
        
        input_item = self.input_texts[self.current_input_index]
        return {
            "success": True,
            "input_id": input_item["id"],
            "text": input_item["text"],
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "index": self.current_input_index,
            "total": len(self.input_texts)
        }

    def _get_all_inputs(self) -> Dict[str, Any]:
        """Get all input texts."""
        return {
            "success": True,
            "inputs": self.input_texts,
            "total": len(self.input_texts)
        }

    def _judge_decision(
        self,
        decision: str,
        reasoning: str,
        final_translation: Optional[str] = None,
        input_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Judge makes a decision: finalize or request another round.

        Args:
            decision: "finalize" or "another_round"
            reasoning: Explanation of decision
            final_translation: Final translation (if finalizing)
            input_id: Input text ID (optional)

        Returns:
            Dict with decision result
        """
        if not input_id:
            if self.current_input_index < len(self.input_texts):
                input_id = self.input_texts[self.current_input_index]["id"]
                source_text = self.input_texts[self.current_input_index]["text"]
            else:
                return {
                    "success": False,
                    "error": "No input text available"
                }
        else:
            input_item = next((item for item in self.input_texts if item["id"] == input_id), None)
            if not input_item:
                return {
                    "success": False,
                    "error": f"Input ID {input_id} not found"
                }
            source_text = input_item["text"]
        
        if decision == "finalize":
            if not final_translation:
                return {
                    "success": False,
                    "error": "final_translation is required when decision is 'finalize'"
                }
            
            # Store final translation
            self.final_translations[input_id] = final_translation
            
            # Evaluate final translation
            eval_scores = {}
            if self.evaluation_enabled:
                eval_scores = self._evaluate_translation(
                    source=source_text,
                    translation=final_translation
                )
            
            self.logger.info(
                f"Final translation for {input_id}: "
                f"COMET={eval_scores.get('comet_score', 'N/A')}, "
                f"BLEURT={eval_scores.get('bleurt_score', 'N/A')}"
            )
            
            # Move to next input
            self.current_input_index += 1
            self.current_round = 0
            
            return {
                "success": True,
                "decision": "finalized",
                "final_translation": final_translation,
                "input_id": input_id,
                "reasoning": reasoning,
                "evaluation": eval_scores,
                "next_input_available": self.current_input_index < len(self.input_texts)
            }
        
        elif decision == "another_round":
            # Increment round counter
            self.current_round += 1
            
            return {
                "success": True,
                "decision": "another_round",
                "new_round": self.current_round,
                "reasoning": reasoning,
                "message": f"Round {self.current_round} requested. Debaters should submit new translations."
            }
        
        else:
            return {
                "success": False,
                "error": f"Invalid decision: {decision}. Must be 'finalize' or 'another_round'"
            }

    def apply_action(
        self, 
        agent_id: Optional[str], 
        action_name: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Override to track current agent for evaluation.

        Args:
            agent_id: ID of agent performing action
            action_name: Name of action
            arguments: Action arguments

        Returns:
            Action result
        """
        self.current_agent_id = agent_id
        result = super().apply_action(agent_id, action_name, arguments)
        return result

    def get_state(self) -> Dict[str, Any]:
        """
        Get current environment state.

        Returns:
            Dict with current state
        """
        state = super().get_state()
        state.update({
            "translations": {
                agent_id: [
                    {
                        "translation": t["translation"],
                        "rationale": t.get("rationale", ""),  # Include rationale
                        "input_id": t["input_id"],
                        "round": t["round"],
                        "evaluation": t.get("evaluation", {})
                    }
                    for t in translations
                ]
                for agent_id, translations in self.translations.items()
            },
            "final_translations": self.final_translations,
            "current_round": self.current_round,
            "current_input_index": self.current_input_index,
            "total_inputs": len(self.input_texts)
        })
        return state

