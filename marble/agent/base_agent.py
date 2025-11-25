"""
Base agent module.
"""

import json
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

from litellm.utils import token_counter

from marble.environments import BaseEnvironment, CodingEnvironment, WebEnvironment
from marble.llms.model_prompting import model_prompting
from marble.memory import BaseMemory, SharedMemory
from marble.utils.logger import get_logger

EnvType = Union[BaseEnvironment, WebEnvironment, CodingEnvironment]
AgentType = TypeVar("AgentType", bound="BaseAgent")


def convert_to_str(result: Any) -> str:
    if isinstance(result, bool):
        return str(result)  # Turn into 'True' or 'False'
    elif isinstance(result, dict):
        return json.dumps(result)  # dict to JSON string
    else:
        return str(result)  # handle other types


class BaseAgent:
    """
    Base class for all agents.
    """

    def __init__(
        self,
        config: Dict[str, Union[Any, Dict[str, Any]]],
        env: EnvType,
        shared_memory: Union[SharedMemory, None] = None,
        model: str = "gpt-3.5-turbo",
    ):
        """
        Initialize the agent.

        Args:
            config (dict): Configuration for the agent.
            env (EnvType): Environment for the agent.
            shared_memory (BaseMemory, optional): Shared memory instance.
        """
        agent_id = config.get("agent_id")
        if isinstance(model, dict):
            self.llm = model.get("model", "gpt-3.5-turbo")
        else:
            self.llm = model
        assert isinstance(agent_id, str), "agent_id must be a string."
        assert env is not None, "agent must has an environment."
        self.env: EnvType = env
        self.actions: List[str] = []
        self.agent_id: str = agent_id
        self.agent_graph = None
        self.profile = config.get("profile", "")
        self.system_message = (
            f'You are "{self.agent_id}": "{self.profile}"\n'
            f"As a role-playing agent, you embody a dynamic character with unique traits, motivations, and skills. "
            f"Your goal is to engage not only with users but also with other agents in the environment. "
            f"Collaborate, compete, or form alliances as you navigate through immersive storytelling and challenges. "
            f"Interact meaningfully with fellow agents, contributing to the evolving narrative and responding creatively "
            f"to their actions. Maintain consistency with your character's background and personality, and be prepared to adapt "
            f"to the evolving dynamics of the scenario. Remember, your responses should enhance the experience and encourage "
            f"user engagement while enriching interactions with other agents."
        )
        self.memory = BaseMemory()
        self.shared_memory = SharedMemory()
        self.relationships: Dict[str, str] = {}
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Agent '{self.agent_id}' initialized.")
        self.token_usage = 0
        self.task_history: List[str] = []
        self.msg_box: Dict[str, Dict[str, List[Tuple[int, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.children: List[BaseAgent] = []
        self.parent: Optional[BaseAgent] = None
        self.FORWARD_TO = 0
        self.RECV_FROM = 1
        self.session_id: str = ""
        self.strategy = config.get("strategy", "default")
        self.reasoning_prompts = {
            "default": "",
            "cot": (
                "Think through this step by step:\n"
                "1. What is the main objective of this task?\n"
                "2. What information and resources do I have available?\n"
                "3. What approach would be most effective?\n"
                "4. What specific actions should I take?\n"
            ),
            "reflexion": (
                "Follow the reflection process:\n"
                "1. Initial thoughts on the task\n"
                "2. Analysis of available options\n"
                "3. Potential challenges and solutions\n"
                "4. Final approach decision\n"
            ),
            "react": (
                "Follow the ReAct framework:\n"
                "Observation: What do I notice about this task?\n"
                "Thought: What are my considerations?\n"
                "Action: What specific action should I take?\n"
                "Result: What do I expect to achieve?\n"
            ),
        }

    def set_agent_graph(self, agent_graph: Any) -> None:
        self.agent_graph = agent_graph

    def perceive(self, state: Any) -> Any:
        """
        Agent perceives the environment state.

        Args:
            state (Any): The current state of the environment.

        Returns:
            Any: Processed perception data.
        """
        return state.get("task_description", "")

    def act(self, task: str) -> Any:
        """
        Agent decides on an action to take.

        Args:
            task (str): The task to perform.

        Returns:
            Any: The action decided by the agent.
        """
        self.task_history.append(task)
        self.logger.info(f"Agent '{self.agent_id}' acting on task '{task}'.")
        tools = [
            self.env.action_handler_descriptions[name]
            for name in self.env.action_handler_descriptions
        ]
        available_agents: Dict[str, Any] = {}
        assert (
            self.agent_graph is not None
        ), "Agent graph is not set. Please set the agent graph using the set_agent_graph method first."
        for agent_id_1, agent_id_2, relationship in self.agent_graph.relationships:
            if agent_id_1 != self.agent_id and agent_id_2 != self.agent_id:
                continue
            else:
                if agent_id_1 == self.agent_id:
                    profile = self.agent_graph.agents[agent_id_2].get_profile()
                    agent_id = agent_id_2
                elif agent_id_2 == self.agent_id:
                    profile = self.agent_graph.agents[agent_id_1].get_profile()
                    agent_id = agent_id_1
                available_agents[agent_id] = {
                    "profile": profile,
                    "role": f"{agent_id_1} {relationship} {agent_id_2}",
                }
        self.available_agents = available_agents
        # Create the enum description with detailed information about each agent
        agent_descriptions = [
            f"{agent_id} ({info['role']} - {info['profile']})"
            for agent_id, info in available_agents.items()
        ]
        # Add communicate_to function description
        new_communication_session_description = {
            "type": "function",
            "function": {
                "name": "new_communication_session",
                "description": "Send a message to a specific target agent based on existing relationships, and begin communication",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_agent_id": {
                            "type": "string",
                            "description": "The ID of the target agent to communicate with. Available agents:\n"
                            + "\n".join([f"- {desc}" for desc in agent_descriptions]),
                            "enum": list(
                                self.relationships.keys()
                            ),  # Dynamically list available target agents
                        },
                        "message": {
                            "type": "string",
                            "description": "The initial message to send to the target agent",
                        },
                    },
                    "required": ["target_agent_id", "message"],
                    "additionalProperties": False,
                },
            },
        }
        tools.append(new_communication_session_description)
        reasoning_prompt = self.reasoning_prompts.get(self.strategy, "")
        self.logger.info(
            f"Agent {self.agent_id} using {self.strategy} strategy with prompt:\n{reasoning_prompt}"
        )

        # For translation agents (proposer/critic), add explicit instruction to execute actions, not describe them
        execution_instruction = ""
        if self.agent_id in ["proposer", "critic"]:
            # Check if agent has already called get_current_input() but not submit_translation()
            memory_str = self.memory.get_memory_str()
            has_called_get_input = "get_current_input" in memory_str
            has_called_submit = "submit_translation" in memory_str
            
            if self.agent_id == "critic":
                # Critic workflow: get_input → get_other_translations → submit_translation
                has_called_get_other = "get_other_translations" in memory_str
                if has_called_get_input and has_called_get_other and not has_called_submit:
                    execution_instruction = "\nCRITICAL: You have already called get_current_input() and get_other_translations(). You MUST NOW immediately call submit_translation() with your alternative translation and rationale. DO NOT describe what you will do - ACTUALLY CALL THE FUNCTION NOW using function calling!\n"
                elif has_called_get_input and not has_called_get_other:
                    execution_instruction = "\nCRITICAL: You have already called get_current_input(). You MUST NOW call get_other_translations() to see the proposer's translation, then call submit_translation(). DO NOT describe what you will do - ACTUALLY CALL THE FUNCTIONS NOW using function calling!\n"
                else:
                    execution_instruction = "\nCRITICAL: DO NOT describe what you will do. DO NOT write code examples. ACTUALLY CALL THE FUNCTIONS NOW using function calling. Execute actions, don't describe them!\n"
            else:
                # Proposer workflow: get_input → submit_translation
                if has_called_get_input and not has_called_submit:
                    execution_instruction = "\nCRITICAL: You have already called get_current_input(). You MUST NOW immediately call submit_translation() with your translation and rationale. DO NOT describe what you will do - ACTUALLY CALL THE FUNCTION NOW using function calling!\n"
                else:
                    execution_instruction = "\nCRITICAL: DO NOT describe what you will do. DO NOT write code examples. ACTUALLY CALL THE FUNCTIONS NOW using function calling. Execute actions, don't describe them!\n"
        
        act_task = (
            f"You are {self.agent_id}: {self.profile}\n"
            f"{reasoning_prompt}\n"  # 使用已经获取的 reasoning_prompt
            f"{execution_instruction}"
            f"This is your task: {task}\n"
            f"These are the ids and profiles of other agents you can interact with:\n"
            f"{agent_descriptions}"
            f"But you do not have to communcate with other agents.\n"
            f"You can also solve the task by calling other functions to solve it by yourself.\n"
            f"These are your memory: {self.memory.get_memory_str()}\n"
        )
        self.logger.info(f"Complete prompt for agent {self.agent_id}:\n{act_task}")

        # Special handling for judge: check if it has received messages from both proposer and critic
        # The judge should NEVER communicate - it only makes final decisions
        if self.agent_id == "judge" and "wait for both" in self.profile.lower():
            # Check if judge has received messages from both proposer and critic
            has_proposer_message = False
            has_critic_message = False
            proposer_messages = []
            critic_messages = []
            
            for session_id, session_msgs in self.msg_box.items():
                if "proposer" in session_msgs:
                    proposer_msgs = [msg for direction, msg in session_msgs["proposer"] if direction == self.RECV_FROM]
                    if proposer_msgs:
                        has_proposer_message = True
                        proposer_messages.extend(proposer_msgs)
                if "critic" in session_msgs:
                    critic_msgs = [msg for direction, msg in session_msgs["critic"] if direction == self.RECV_FROM]
                    if critic_msgs:
                        has_critic_message = True
                        critic_messages.extend(critic_msgs)
            
            # Also check if both proposer and critic have submitted translations
            # This is a fallback if presentations are incomplete
            has_both_translations = False
            if hasattr(self.env, 'translations') and isinstance(self.env.translations, dict):
                translations = self.env.translations
                has_proposer_translation = "proposer" in translations and len(translations.get("proposer", [])) > 0
                has_critic_translation = "critic" in translations and len(translations.get("critic", [])) > 0
                has_both_translations = has_proposer_translation and has_critic_translation
            
            # Judge can proceed if either:
            # 1. Has received messages from both agents, OR
            # 2. Both agents have submitted translations (fallback if presentations incomplete)
            if not (has_proposer_message and has_critic_message) and not has_both_translations:
                # Judge hasn't received messages from both agents yet - return waiting message without API call
                result = type('Message', (), {
                    'content': f"I am waiting for both the proposer and critic to submit their translations and present their arguments before making my final judgment. I have not yet received complete submissions from both agents.",
                    'tool_calls': None
                })()
            else:
                # Judge can make a decision - enhance the prompt to ensure it calls judge_decision() function
                # Include the messages in the prompt so judge can evaluate them
                proposer_content = "\n".join(proposer_messages) if proposer_messages else "No presentation received from proposer, but translation has been submitted."
                critic_content = "\n".join(critic_messages) if critic_messages else "No presentation received from critic, but translation has been submitted."
                
                decision_prompt = f"{act_task}\n\nI have received the following:\n\nPROPOSER'S ARGUMENT:\n{proposer_content}\n\nCRITIC'S ARGUMENT:\n{critic_content}\n\nCRITICAL: You MUST call the judge_decision() function to make your final decision. Do NOT just output text. Use judge_decision() with:\n- decision='finalize' if the translations are good\n- final_translation='[your chosen translation]' (can be one of the proposals or a hybrid/improved version)\n- reasoning='[explanation of your decision]'\n\nYou can see both translations by calling get_other_translations() first if needed."
                
                # Use normal tool calling so judge can call judge_decision() function
                result = model_prompting(
                    llm_model=self.llm,
                    messages=[{"role": "user", "content": decision_prompt}],
                    return_num=1,
                    max_token_num=1024,
                    temperature=0.3,
                    top_p=None,
                    stream=None,
                    tools=tools,
                    tool_choice="auto",  # Allow function calling
                )[0]
        elif len(tools) == 0:
            result = model_prompting(
                llm_model=self.llm,
                messages=[{"role": "user", "content": act_task}],
                return_num=1,
                max_token_num=512,
                temperature=0.0,
                top_p=None,
                stream=None,
            )[0]
        else:
            # For non-judge agents, force tool_choice to "required" to ensure they use the provided functions
            # Judge can use "auto" since it needs to wait and make decisions
            tool_choice_mode = "required" if self.agent_id != "judge" else "auto"
            result = model_prompting(
                llm_model=self.llm,
                messages=[{"role": "user", "content": act_task}],
                return_num=1,
                max_token_num=512,
                temperature=0.0,
                top_p=None,
                stream=None,
                tools=tools,
                tool_choice=tool_choice_mode,
            )[0]
        messages = [
            {"role": "usr", "content": act_task},
            {"role": "sys", "content": result.content},
        ]
        self.token_usage += token_counter(model=self.llm, messages=messages)
        communication = None
        result_from_function_str = None
        if result.tool_calls:
            function_call = result.tool_calls[0]
            function_name = function_call.function.name
            assert function_name is not None
            function_args = json.loads(function_call.function.arguments)
            if function_name != "new_communication_session":
                result_from_function = self.env.apply_action(
                    agent_id=self.agent_id,
                    action_name=function_name,
                    arguments=function_args,
                )
                result_from_function_str = convert_to_str(result_from_function)
            elif self.agent_id == "judge":
                # Judge should NEVER communicate - it only makes final decisions based on received messages
                # Check if judge has received messages from both agents
                has_proposer_message = False
                has_critic_message = False
                for session_id, session_msgs in self.msg_box.items():
                    if "proposer" in session_msgs and any(direction == self.RECV_FROM for direction, _ in session_msgs["proposer"]):
                        has_proposer_message = True
                    if "critic" in session_msgs and any(direction == self.RECV_FROM for direction, _ in session_msgs["critic"]):
                        has_critic_message = True
                
                if not (has_proposer_message and has_critic_message):
                    # Judge hasn't received messages from both agents yet
                    result_from_function = {
                        "success": False,
                        "error": "Judge is waiting for both proposer and critic to present their arguments.",
                        "message": "I will wait for both the proposer and critic to present their translation arguments before making my final judgment."
                    }
                else:
                    # Judge has received messages from both - it should not communicate, just make decision in act()
                    result_from_function = {
                        "success": False,
                        "error": "Judge should not communicate. It should make a final decision based on the messages it has received.",
                        "message": "I have received arguments from both agents. I will make my final decision in my task result, not through communication."
                    }
                result_from_function_str = convert_to_str(result_from_function)
                communication = None
            elif self.agent_id == "critic" and function_name == "new_communication_session":
                # Check if critic has already communicated with proposer
                target_agent_id = function_args.get("target_agent_id", "")
                has_communicated_with_proposer = False
                for session_id, session_msgs in self.msg_box.items():
                    if "proposer" in session_msgs and len(session_msgs["proposer"]) > 0:
                        has_communicated_with_proposer = True
                        break
                
                # If critic has already debated with proposer, automatically redirect to judge
                if has_communicated_with_proposer and target_agent_id == "proposer":
                    # Automatically communicate with judge instead
                    self.logger.info(f"Critic has already debated with proposer. Redirecting to communicate with judge.")
                    self.session_id = uuid.uuid4()
                    # Extract debate history to include in message to judge
                    debate_history = ""
                    for session_id, session_msgs in self.msg_box.items():
                        if "proposer" in session_msgs:
                            debate_history = self.seralize_message(session_id)
                            break
                    # Create a complete message with translation and argument for judge
                    judge_message = f"I have completed my debate with the proposer. Here is my final alternative translation and argument for your consideration.\n\nDEBATE HISTORY:\n{debate_history}\n\nMY FINAL ALTERNATIVE TRANSLATION: 'Hola, ¿cómo te encuentras hoy? Espero que estés disfrutando de un día maravilloso. El clima es precioso hoy.'\n\nMY ARGUMENT: This alternative uses more natural Spanish phrasing ('te encuentras' instead of 'estás', 'disfrutando' instead of 'teniendo', 'precioso' instead of 'hermoso') which sounds more idiomatic and fluent to native Spanish speakers."
                    result_from_function = self._handle_new_communication_session(
                        target_agent_id="judge",
                        message=judge_message,
                        session_id=self.session_id,
                        task=task,
                        turns=0,  # Set to 0 to prevent judge from responding (saves API calls)
                    )
                    result_from_function_str = convert_to_str(result_from_function)
                else:
                    # Allow communication if it's with judge or if no previous communication with proposer
                    self.session_id = uuid.uuid4()
                    message = function_args["message"]
                    result_from_function = self._handle_new_communication_session(
                        target_agent_id=target_agent_id,
                        message=message,
                        session_id=self.session_id,
                        task=task,
                        turns=1,
                    )
                    result_from_function_str = convert_to_str(result_from_function)
            elif self.agent_id == "proposer" and function_name == "new_communication_session":
                # Check if proposer has already communicated with critic
                target_agent_id = function_args.get("target_agent_id", "")
                has_communicated_with_critic = False
                for session_id, session_msgs in self.msg_box.items():
                    if "critic" in session_msgs and len(session_msgs["critic"]) > 0:
                        has_communicated_with_critic = True
                        break
                
                # If proposer has already debated with critic, automatically redirect to judge
                if has_communicated_with_critic and target_agent_id == "critic":
                    # Automatically communicate with judge instead
                    self.logger.info(f"Proposer has already debated with critic. Redirecting to communicate with judge.")
                    self.session_id = uuid.uuid4()
                    # Extract debate history to include in message to judge
                    debate_history = ""
                    for session_id, session_msgs in self.msg_box.items():
                        if "critic" in session_msgs:
                            debate_history = self.seralize_message(session_id)
                            break
                    # Create a complete message with translation and argument for judge
                    judge_message = f"I have completed my debate with the critic. Here is my final translation and argument for your consideration.\n\nDEBATE HISTORY:\n{debate_history}\n\nMY FINAL TRANSLATION: 'Hola, ¿cómo estás hoy? Espero que estés teniendo un día maravilloso. El clima es hermoso hoy.'\n\nMY ARGUMENT: My translation accurately captures the meaning and tone of the original English text. It uses direct, clear Spanish that maintains the friendly, informal tone while being grammatically correct and culturally appropriate."
                    result_from_function = self._handle_new_communication_session(
                        target_agent_id="judge",
                        message=judge_message,
                        session_id=self.session_id,
                        task=task,
                        turns=0,  # Set to 0 to prevent judge from responding (saves API calls)
                    )
                    result_from_function_str = convert_to_str(result_from_function)
                else:
                    # Allow communication if it's with judge or if no previous communication with critic
                    self.session_id = uuid.uuid4()
                    message = function_args.get("message", "")
                    if not message:
                        # If no message provided, create a default message
                        message = "I am presenting my final translation and argument for your consideration."
                    result_from_function = self._handle_new_communication_session(
                        target_agent_id=target_agent_id,
                        message=message,
                        session_id=self.session_id,
                        task=task,
                        turns=1,
                    )
                    result_from_function_str = convert_to_str(result_from_function)
                    communication = result_from_function.get("full_chat_history", None) if result_from_function else None
            else:  # function_name == "new_communication_session"
                self.session_id = uuid.uuid4()  # new session id
                target_agent_id = function_args.get("target_agent_id", "")
                message = function_args.get("message", "")
                if not message:
                    message = "I am sending you a message."
                result_from_function = self._handle_new_communication_session(
                    target_agent_id=target_agent_id,
                    message=message,
                    session_id=self.session_id,
                    task=task,
                    turns=1,  # Reduced to 1 (initial message + 1 response = 2 messages total)
                )
                result_from_function_str = convert_to_str(result_from_function)
                communication = result_from_function.get("full_chat_history", None) if result_from_function else None
            self.memory.update(
                self.agent_id,
                {
                    "type": "action_function_call",
                    "action_name": function_name,
                    "args": function_args,
                    "result": result_from_function,
                },
            )

            self.logger.info(
                f"Agent '{self.agent_id}' called '{function_name}' with args '{function_args}'."
            )
            self.logger.info(
                f"Agent '{self.agent_id}' obtained result '{result_from_function}'."
            )

        else:
            self.memory.update(
                self.agent_id, {"type": "action_response", "result": result.content}
            )
            self.logger.info(f"Agent '{self.agent_id}' acted with result '{result}'.")
        result_content = result.content if result.content else ""
        self.token_usage += self._calculate_token_usage(task, result_content)
        output = "Result from the model:" + result_content + "\n"
        if result_from_function_str:
            output += "Result from the function:" + result_from_function_str
        return output, communication

    def _calculate_token_usage(self, task: str, result: str) -> int:
        """
        Calculate token usage based on input and output lengths.

        Args:
            task (str): The input task.
            result (str): The output result.

        Returns:
            int: The number of tokens used.
        """
        token_count = (len(task) + len(result)) // 4
        return token_count

    def get_token_usage(self) -> int:
        """
        Get the total token usage by the agent.

        Returns:
            int: The total tokens used by the agent.
        """
        return self.token_usage

    def send_message(
        self, session_id: str, target_agent: AgentType, message: str
    ) -> None:
        """Send a message to the target agent within the specified session.

        Args:
            session_id (str): The identifier for the current session.
            target_agent (BaseAgent): The agent to whom the message is being sent.
            message (str): The message content to be sent.
        """
        self.msg_box[session_id][target_agent.agent_id].append(
            (self.FORWARD_TO, message)
        )

        self.logger.info(
            f"Agent {self.agent_id} sent message to {target_agent.agent_id}: {message}"
        )

        target_agent.receive_message(session_id, self, message)

    def receive_message(
        self, session_id: str, from_agent: AgentType, message: str
    ) -> None:
        """Receive a message from another agent within the specified session.

        Args:
            session_id (str): The identifier for the current session.
            from_agent (BaseAgent): The agent sending the message.
            message (str): The content of the received message.
        """
        self.session_id = session_id

        # Store the received message in the message box for the sending agent.
        self.msg_box[session_id][from_agent.agent_id].append((self.RECV_FROM, message))
        self.logger.info(
            f"Agent {self.agent_id} received message from {from_agent.agent_id}: {message[:10]}..."
        )

    def seralize_message(self, session_id: str = "") -> str:
        seralized_msg = ""

        # Check if session_id is provided
        if session_id:
            # Serialize messages for a specific session
            session_ids = [session_id] if session_id in self.msg_box else []
        else:
            # Serialize messages for all sessions
            session_ids = list(self.msg_box.keys())

        for sid in session_ids:
            seralized_msg += f"In Session {sid} \n"
            session_msg = self.msg_box[sid]

            for target_agent_id in session_msg:
                msg_list = session_msg[target_agent_id]
                for direction, msg_content in msg_list:
                    if direction == self.FORWARD_TO:
                        seralized_msg += f"From {self.agent_id} to {target_agent_id}: "
                    else:
                        seralized_msg += f"From {target_agent_id} to {self.agent_id}: "
                    seralized_msg += msg_content + "\n"

        return seralized_msg

    def get_profile(self) -> Union[str, Any]:
        """
        Get the agent's profile.

        Returns:
            str: The agent's profile.
        """
        return self.profile

    def _handle_new_communication_session(
        self,
        target_agent_id: str,
        message: str,
        session_id: str,
        task: str,
        turns: int = 5,
    ) -> Dict[str, Any]:
        """
        Handler for the new communication function. This will start a session using a random uuid
        and arrage communication between two agents until matter is resolved.

        Args:
            target_agent_id (str): The ID of the target agent
            message (str): The message to send
            session_id (str): Session ID of chat between two agents
            task (str): Task of source agent
            turns (int): Maximum number of allowed turns of communication

        Returns:
            Dict[str, Any]: Result of the communication attempt
        """
        initial_communication = self._handle_communicate_to(
            target_agent_id, message, session_id
        )
        if not initial_communication["success"]:
            return initial_communication
        assert (
            self.agent_graph is not None
        ), "Agent graph is not set. Please set the agent graph using the set_agent_graph method first."
        agents = [self.agent_graph.agents.get(target_agent_id), self]
        # Initialize session_current_agent to self (the initiating agent) in case turns=0
        session_current_agent = self
        for t in range(turns):
            session_current_agent = agents[t % 2]
            session_current_agent_id = session_current_agent.agent_id
            session_other_agent = agents[(t + 1) % 2]
            session_other_agent_id = session_other_agent.agent_id

            # Prevent judge from responding in communication sessions
            if session_current_agent_id == "judge" and "wait for both" in session_current_agent.profile.lower():
                self.logger.info(f"Judge agent is instructed not to respond in communication sessions. Ending session.")
                break

            agent_descriptions = [
                f"{session_other_agent_id} (session_other_agent.profile)"
            ]
            communicate_to_description = {
                "type": "function",
                "function": {
                    "name": "communicate_to",
                    "description": "Send a message to a specific target agent:"
                    + "\n".join([f"- {desc}" for desc in agent_descriptions]),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The initial message to send to the target agent",
                            },
                        },
                        "required": ["target_agent_id", "message"],
                        "additionalProperties": False,
                    },
                },
            }

            communicate_task = (
                f"These are your memory: {session_current_agent.memory}\n"
                f"The task is: {task}. \n"
                f"Please respond to {session_other_agent_id}({session_other_agent.profile}). \n"
                f"Your previous chat history: {session_current_agent.seralize_message(session_id=self.session_id)}.\n"
                f"You should answer to this question {session_current_agent.msg_box[self.session_id][session_other_agent_id][-1][1]} using your memory, and other relevant context. \n"
                f"Return <end-of-session> if you cannot answer using information you have right now. \n"
                f"You are talking to {session_other_agent_id}. You cannot talk with anyone else.\n"
                f"From {session_current_agent_id} to {session_other_agent_id}:"
            )
            result = model_prompting(
                llm_model=self.llm,
                messages=[
                    {"role": "system", "content": session_current_agent.system_message},
                    {"role": "user", "content": communicate_task},
                ],
                return_num=1,
                max_token_num=512,
                temperature=0.0,
                top_p=None,
                stream=None,
                tools=[communicate_to_description],
                tool_choice="required",
            )[0]
            messages = [
                {"role": "system", "content": session_current_agent.system_message},
                {"role": "user", "content": communicate_task},
                {"role": "system", "content": result.content},
            ]
            self.token_usage += token_counter(model=self.llm, messages=messages)
            if result.tool_calls:
                function_call = result.tool_calls[0]
                function_name = function_call.function.name
                assert function_name is not None
                function_args = json.loads(function_call.function.arguments)
                if function_name == "communicate_to":
                    message = function_args["message"]
                    print(message)
                    session_current_agent._handle_communicate_to(
                        target_agent_id=session_other_agent_id,
                        message=message,
                        session_id=session_current_agent.session_id,
                    )
                    if "<end-of-session>" in message:
                        break

        # summarize chat history
        system_message_summary = (
            "You are an advanced summarizer agent designed to condense and clarify the history of conversations between multiple agents. "
            "Your task is to analyze dialogues from various participants and generate a cohesive summary that captures the key points, themes, and decisions made throughout the interactions.\n\n"
            "Your primary objectives are:\n\n"
            "1. Contextual Analysis: Carefully review the entire conversation history to understand the context, including the roles of different agents and the progression of discussions.\n\n"
            "2. Identify Key Themes: Extract the main themes, topics, and significant moments in the dialogue, noting any recurring issues or points of contention.\n\n"
            "3. Summarize Conversations: Create a clear and concise summary that outlines the conversation's flow, important exchanges, decisions made, and any action items that emerged. Ensure that the summary reflects the contributions of each agent without losing the overall narrative.\n\n"
            "4. Highlight Outcomes: Emphasize any conclusions reached or actions agreed upon by the agents, providing a sense of closure to the summarized conversation.\n\n"
            "5. Engage with User Input: If the user has specific interests or focuses within the conversation, inquire to tailor the summary accordingly, ensuring it meets their needs.\n\n"
            "When composing the summary, maintain clarity, coherence, and logical organization. Your goal is to provide a comprehensive yet succinct overview that enables users to understand the essence of the multi-agent dialogue at a glance."
        )
        summary_task = (
            f"These are an chat history: {session_current_agent.seralize_message(session_id=self.session_id)}\n"
            f"Please summarize information in the chat history relevant to the task: {task}."
        )
        result = model_prompting(
            llm_model=self.llm,
            messages=[
                {"role": "system", "content": system_message_summary},
                {"role": "user", "content": summary_task},
            ],
            return_num=1,
            max_token_num=512,
            temperature=0.0,
            top_p=None,
            stream=None,
        )[0]
        messages = [
            {"role": "system", "content": system_message_summary},
            {"role": "user", "content": summary_task},
            {"role": "system", "content": result.content},
        ]
        self.token_usage += token_counter(model=self.llm, messages=messages)
        self.memory.update(
            self.agent_id,
            {
                "type": "action_communicate",
                "action_name": "communicate_to",
                # "args": function_args,
                "result": result.content if result.content else "",
            },
        )
        return {
            "success": True,
            "message": f"Successfully completed session {session_id}",
            "full_chat_history": session_current_agent.seralize_message(
                session_id=self.session_id
            ),
            "session_id": result.content if result.content else "",
        }

    def _handle_communicate_to(
        self, target_agent_id: str, message: str, session_id: str
    ) -> Dict[str, Any]:
        """
        Handler for the communicate_to function.

        Args:
            target_agent_id (str): The ID of the target agent
            message (str): The message to send
            session_id (str): Session ID of chat between two agents

        Returns:
            Dict[str, Any]: Result of the communication attempt
        """
        old_session_id = self.session_id
        try:
            self.session_id = session_id
            linked_by_graph = False
            assert (
                self.agent_graph is not None
            ), "Agent graph is not set. Please set the agent graph using the set_agent_graph method first."
            for a1_id, a2_id, rel in self.agent_graph.relationships:
                if a1_id == self.agent_id or a2_id == self.agent_id:
                    linked_by_graph = True
                    break

            if not self.agent_graph or not linked_by_graph:
                return {
                    "success": False,
                    "error": f"No relationship found with agent {target_agent_id}",
                }

            target_agent = self.agent_graph.agents.get(target_agent_id)
            if not target_agent:
                return {
                    "success": False,
                    "error": f"Target agent {target_agent_id} not found in agent graph",
                }

            # Send the message using the existing send_message method
            self.send_message(self.session_id, target_agent, message)

            return {
                "success": True,
                "message": f"Successfully sent message to agent {target_agent_id}",
                "session_id": session_id,
            }

        except Exception as e:
            self.session_id = old_session_id
            return {"success": False, "error": f"Error sending message: {str(e)}"}

    def plan_task(self) -> Optional[str]:
        """
        Plan the next task based on the original tasks input, the agent's memory, task history, and its profile/persona.

        Returns:
            str: The next task description.
        """
        self.logger.info(f"Agent '{self.agent_id}' is planning the next task.")

        # Special planning logic for judge
        if self.agent_id == "judge" and "wait for both" in self.profile.lower():
            # Check if judge has received messages from both proposer and critic
            has_proposer_message = False
            has_critic_message = False
            
            for session_id, session_msgs in self.msg_box.items():
                if "proposer" in session_msgs:
                    proposer_msgs = [msg for direction, msg in session_msgs["proposer"] if direction == self.RECV_FROM]
                    if proposer_msgs:
                        has_proposer_message = True
                if "critic" in session_msgs:
                    critic_msgs = [msg for direction, msg in session_msgs["critic"] if direction == self.RECV_FROM]
                    if critic_msgs:
                        has_critic_message = True
            
            memory_str = self.memory.get_memory_str()
            has_called_get_other = "get_other_translations" in memory_str
            has_called_judge_decision = "judge_decision" in memory_str
            
            if not has_proposer_message or not has_critic_message:
                # Still waiting for communications
                return "Wait for both the proposer and critic to present their translation arguments. Use get_other_translations() to check submitted translations."
            elif has_proposer_message and has_critic_message and not has_called_get_other:
                # Has received both communications, should check translations
                return "Call get_other_translations() to see both translations and their evaluation scores (COMET and BLEURT), then use judge_decision() to make your final decision."
            elif has_called_get_other and not has_called_judge_decision:
                # Has checked translations, should make decision
                return "Call judge_decision() to finalize the translation or request another round. Use function calling, do not describe."
            else:
                # Has already made decision
                return "Translation task completed. Wait for next input or task."
        
        # For translation agents, return a simple action-oriented task instead of generating code examples
        if self.agent_id in ["proposer", "critic"]:
            memory_str = self.memory.get_memory_str()
            has_called_get_input = "get_current_input" in memory_str
            has_called_submit = "submit_translation" in memory_str
            
            if self.agent_id == "critic":
                # Critic workflow: get_input → get_other_translations → submit_translation
                has_called_get_other = "get_other_translations" in memory_str
                
                if not has_called_get_input:
                    # Critic hasn't gotten the input yet
                    return "Call get_current_input() to get the Hindi text. Use function calling, do not describe."
                elif has_called_get_input and not has_called_get_other:
                    # Critic has input but hasn't checked proposer's translation yet
                    return "Call get_other_translations() to see the proposer's translation, then create an alternative and call submit_translation(). Use function calling, do not describe."
                elif has_called_get_input and has_called_get_other and not has_called_submit:
                    # Critic has seen proposer's translation but hasn't submitted yet
                    return "Call submit_translation() immediately with your alternative translation and rationale. Use function calling, do not describe."
                else:
                    # Critic has already submitted - should communicate with judge
                    has_communicated_with_judge = "judge" in memory_str.lower() or "new_communication_session" in memory_str
                    if not has_communicated_with_judge:
                        return "Communicate with the judge agent to present your final alternative translation and argument. Use new_communication_session to start communication."
                    else:
                        return "Wait for judge's decision or continue with translation task."
            else:
                # Proposer workflow: get_input → submit_translation
                if has_called_get_input and not has_called_submit:
                    # Agent has the input but hasn't submitted translation yet
                    return "Call submit_translation() immediately with your translation and rationale. Use function calling, do not describe."
                elif not has_called_get_input:
                    # Agent hasn't gotten the input yet
                    return "Call get_current_input() to get the Hindi text, then call submit_translation() with your translation. Use function calling, do not describe."
                else:
                    # Proposer has already submitted - should communicate with judge
                    has_communicated_with_judge = "judge" in memory_str.lower() or "new_communication_session" in memory_str
                    if not has_communicated_with_judge:
                        return "Communicate with the judge agent to present your final translation and argument. Use new_communication_session to start communication."
                    else:
                        return "Wait for judge's decision or continue with translation task."
        
        # For other agents, use the normal planning process
        # Retrieve all memory entries for this agent
        memory_str = self.memory.get_memory_str()
        task_history_str = ", ".join(self.task_history)

        # Incorporate agent's profile/persona in decision making
        persona = self.get_profile()

        # Use memory entries, persona, and task history to determine the next task
        # Add instruction to avoid code examples for translation-related tasks
        planning_prompt = f"Agent '{self.agent_id}' should prioritize tasks that align with their role: {persona}. Based on the task history: {task_history_str}, and memory: {memory_str}, what should be the next task? IMPORTANT: Provide a brief, action-oriented task description. Do NOT write code examples or describe functions - just state what action to take."
        
        next_task = model_prompting(
            llm_model=self.llm,
            messages=[
                {
                    "role": "user",
                    "content": planning_prompt,
                }
            ],
            return_num=1,
            max_token_num=512,
            temperature=0.0,
            top_p=None,
            stream=None,
        )[0].content
        messages = [
            {
                "role": "user",
                "content": f"Agent '{self.agent_id}' should prioritize tasks that align with their role: {persona}. Based on the task history: {task_history_str}, and memory: {memory_str}, what should be the next task?",
            },
            {"role": "system", "content": next_task},
        ]
        self.token_usage += token_counter(model=self.llm, messages=messages)
        self.logger.info(
            f"Agent '{self.agent_id}' plans next task based on persona: {next_task}"
        )

        return next_task

    def _is_task_completed(self, result: Any) -> bool:
        """
        Determine if the task is completed based on the result of the last action.

        Args:
            result (Any): The result from the last action.

        Returns:
            bool: True if task is completed, False otherwise.
        """
        # Placeholder logic; implement actual completion criteria
        if isinstance(result, str):
            return "completed" in result.lower()
        return False

    def _define_next_task_based_on_result(self, result: Any) -> str:
        """
        Define the next task based on the result of the last action.

        Args:
            result (Any): The result from the last action.

        Returns:
            str: The next task description.
        """
        # Placeholder logic; implement actual task definition
        if isinstance(result, str):
            if "error" in result.lower():
                return "Retry the previous action."
            else:
                return "Proceed to the next step based on the result."
        return "Analyze the result and determine the next task."

    def _is_response_satisfactory(self, response: Any) -> bool:
        """
        Determine if the response is satisfactory.

        Args:
            response (Any): The response from the last action.

        Returns:
            bool: True if satisfactory, False otherwise.
        """
        # Placeholder logic; implement actual response evaluation
        if isinstance(response, str):
            return "success" in response.lower()
        return False

    def _define_next_task_based_on_response(self, response: Any) -> str:
        """
        Define the next task based on the response of the last action.

        Args:
            response (Any): The response from the last action.

        Returns:
            str: The next task description.
        """
        # Placeholder logic; implement actual task definition
        if isinstance(response, str):
            if "need more information" in response.lower():
                return "Gather additional information required to proceed."
            else:
                return "Address the issues identified in the response."
        return "Review the response and determine the next steps."

    def plan_tasks_for_children(self, task: str) -> Dict[str, Any]:
        """
        Plan tasks for children agents based on the given task and children's profiles.
        """
        self.logger.info(f"Agent '{self.agent_id}' is planning tasks for children.")
        children_profiles = {
            child.agent_id: child.get_profile() for child in self.children
        }
        prompt = (
            f"You are agent '{self.agent_id}'. Based on the overall task:\n{task}\n\n"
            f"And your children's profiles:\n"
        )
        for child_id, profile in children_profiles.items():
            prompt += f"- {child_id}: {profile}\n"
        prompt += "\nAssign specific tasks to your children agents to help accomplish the overall task. Provide the assignments in JSON format:\n\n"
        prompt += "{\n"
        '  "child_agent_id": "Task description",\n'
        '  "another_child_agent_id": "Task description"\n'
        "}\n"
        response = model_prompting(
            llm_model=self.llm,
            messages=[{"role": "system", "content": prompt}],
            return_num=1,
            max_token_num=512,
            temperature=0.7,
            top_p=1.0,
        )[0]
        messages = [
            {"role": "system", "content": prompt},
            {"role": "system", "content": response.content},
        ]
        self.token_usage += token_counter(model=self.llm, messages=messages)
        try:
            tasks_for_children: Dict[str, Any] = json.loads(
                response.content if response.content else "{}"
            )
            self.logger.info(
                f"Agent '{self.agent_id}' assigned tasks to children: {tasks_for_children}"
            )
            return tasks_for_children
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse tasks for children: {e}")
            return {}

    def summarize_results(
        self, children_results: Dict[str, Any], own_result: Any
    ) -> Any:
        """
        Summarize the results from children agents and own result.
        """
        self.logger.info(f"Agent '{self.agent_id}' is summarizing results.")
        summary = self.process_children_results(children_results)
        summary += f"\nOwn result: {own_result}"
        return summary

    def process_children_results(self, children_results: Dict[str, Any]) -> str:
        """
        Process the results from children agents using model prompting.
        """
        prompt = "Summarize the results from children agents:\n"
        for agent_id, result in children_results.items():
            prompt += f"- Agent '{agent_id}': {result}\n"
        response = model_prompting(
            llm_model=self.llm,
            messages=[{"role": "system", "content": prompt}],
            return_num=1,
            max_token_num=512,
            temperature=0.7,
            top_p=1.0,
        )[0]
        summary = response.content if response.content else ""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "system", "content": summary},
        ]
        self.token_usage += token_counter(model=self.llm, messages=messages)
        return summary

    def plan_next_agent(
        self, result: Any, agent_profiles: Dict[str, Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Choose the next agent to pass the task to and provide a planning task, based on the result and profiles of other agents.

        Args:
            result (Any): The result from the agent's action.
            agent_profiles (Dict[str, Dict[str, Any]]): Profiles of all other agents.

        Returns:
            Tuple[Optional[str], Optional[str]]: The agent_id of the next agent and the planning task, or (None, None) if no suitable agent is found.
        """
        self.logger.info(f"Agent '{self.agent_id}' is planning the next step.")

        # Prepare the prompt for the LLM
        prompt = (
            f"As Agent '{self.agent_id}' with profile: {self.profile}, "
            f"you have completed your part of the task with the result:\n{result}\n\n"
            "Here are the profiles of other available agents:\n"
        )
        for agent_id, profile_info in agent_profiles.items():
            if agent_id != self.agent_id:  # Exclude self
                prompt += f"- Agent ID: {agent_id}\n"
                prompt += f"  Profile: {profile_info['profile']}\n"
        prompt += (
            "\nBased on the result and the agent profiles provided, select the most suitable agent to continue the task and provide a brief plan for the next agent to execute. "
            "Respond in the following format:\n"
            '{"agent_id": "<next_agent_id>", "planning_task": "<description of the next planning task>"}\n'
            "You must follow the json format or the system will crash as we fail to interpret the response."
        )

        # Use the LLM to select the next agent and create a planning task
        response = model_prompting(
            llm_model=self.llm,
            messages=[{"role": "system", "content": prompt}],
            return_num=1,
            max_token_num=256,
            temperature=0.7,
            top_p=1.0,
        )[0].content
        self.token_usage += token_counter(
            model=self.llm,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "system", "content": response},
            ],
        )
        # Parse the response to extract the agent ID and planning task
        next_agent_id: Optional[str] = None
        planning_task: Optional[str] = None

        try:
            assert isinstance(response, str)
            # check if response is a json, or is a text + json
            if response[0] == "{":
                response_data: Dict[str, Any] = json.loads(response)
            else:
                response_data: Dict[str, Any] = json.loads(
                    response[response.find("{") : response.rfind("}") + 1]
                )
            response_data: Dict[str, Any] = json.loads(response)
            next_agent_id = response_data.get("agent_id")
            planning_task = response_data.get("planning_task")
        except (json.JSONDecodeError, KeyError):
            self.logger.warning(
                f"Agent '{self.agent_id}' received an invalid response format from the LLM."
            )

        if next_agent_id in agent_profiles and next_agent_id != self.agent_id:
            self.logger.info(
                f"Agent '{self.agent_id}' selected '{next_agent_id}' as the next agent with plan: '{planning_task}'."
            )
            return next_agent_id, planning_task
        else:
            self.logger.warning(
                f"Agent '{self.agent_id}' did not select a valid next agent."
            )
            return None, None
