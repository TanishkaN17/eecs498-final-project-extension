# marble/engine/engine.py

"""
The core engine module that coordinates agents within the environment.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Union

from marble.agent import BaseAgent
from marble.configs.config import Config
from marble.engine.engine_planner import EnginePlanner
from marble.environments import (
    BaseEnvironment,
    CodingEnvironment,
    DBEnvironment,
    MinecraftEnvironment,
    ResearchEnvironment,
    TranslationEnvironment,
    WebEnvironment,
    WorldSimulationEnvironment,
)
from marble.evaluator.evaluator import Evaluator
from marble.graph.agent_graph import AgentGraph
from marble.memory.base_memory import BaseMemory
from marble.memory.shared_memory import SharedMemory
from marble.utils.logger import get_logger

EnvType = Union[
    BaseEnvironment,
    WebEnvironment,
    ResearchEnvironment,
    WorldSimulationEnvironment,
    MinecraftEnvironment,
    DBEnvironment,
    CodingEnvironment,
    TranslationEnvironment,
]
AgentType = Union[BaseAgent]


class Engine:
    """
    The Engine class orchestrates the simulation, coordinating agents and the environment.
    """

    def _read_code_from_file(self, file_path: str) -> str:
        """
        Read code from a specified file path.

        Args:
            file_path (str): File path

        Returns:
            str: File content
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except IOError as e:
            self.logger.error(f"Failed to read code from {file_path}: {e}")
            return ""

    def __init__(self, config: Config):
        """
        Initialize the Engine with the given configuration.

        Args:
            config (Config): Configuration parameters.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = config
        self.planning_method = config.engine_planner.get("planning_method", "naive")
        # Initialize Environment
        self.environment = self._initialize_environment(config.environment)
        # Initialize Agents
        self.agents = self._initialize_agents(config.agents)
        # Initialize AgentGraph
        self.graph = AgentGraph(self.agents, config)
        for agent in self.agents:
            agent.set_agent_graph(self.graph)
        # Initialize Memory
        self.memory = self._initialize_memory(config.memory)
        # Initialize Evaluator
        self.evaluator = Evaluator(metrics_config=config.metrics)
        self.task = config.task.get("content", "")
        self.output_format = config.task.get(
            "output_format",
            "You are free to define your own output format to answer the task properly.",
        )
        self.coordinate_mode = config.coordination_mode
        # Initialize EnginePlanner
        self.planner = EnginePlanner(
            agent_graph=self.graph,
            memory=self.memory,
            config=config.engine_planner,
            task=self.task,
            model=config.llm,
        )
        self.max_iterations = config.environment.get("max_iterations", 10)
        self.current_iteration = 0

        self.logger.info("Engine initialized.")

    def _initialize_environment(self, env_config: Dict[str, Any]) -> BaseEnvironment:
        """
        Initialize the environment based on configuration.

        Args:
            env_config (dict): Environment configuration.

        Returns:
            BaseEnvironment: An instance of the environment.

        Raises:
            ValueError: If the environment type is not supported.
        """
        env_type = env_config.get("type")

        if env_type == "Web":
            env1 = WebEnvironment(name="Web Environment", config=env_config)
            return env1
        elif env_type == "Base":
            env2 = BaseEnvironment(name="Base Environment", config=env_config)
            return env2
        elif env_type == "Research":
            env3 = ResearchEnvironment(name="Research Environment", config=env_config)
            return env3
        elif env_type == "Coding":
            env4 = CodingEnvironment(name="Coding Environment", config=env_config)
            return env4
        elif env_type == "WorldSimulation":
            env4 = WorldSimulationEnvironment(
                name="World Simulation Environment", config=env_config
            )
            return env4
        elif env_type == "Minecraft":
            env5 = MinecraftEnvironment(name="Minecraft Environment", config=env_config)
            return env5
        elif env_type == "DB":
            env6 = DBEnvironment(name="DB Environment", config=env_config)
            return env6
        elif env_type == "Translation":
            env7 = TranslationEnvironment(name="Translation Environment", config=env_config)
            return env7
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")

    def _initialize_agents(
        self, agent_configs: List[Dict[str, Any]]
    ) -> List[BaseAgent]:
        """
        Initialize agents based on configurations.

        Args:
            agent_configs (List[dict]): List of agent configurations.

        Returns:
            List[BaseAgent]: List of agent instances.
        """
        agents = []
        llm = self.config.llm
        for agent_config in agent_configs:
            agent_llm = agent_config.get(
                "llm", llm
            )  # use agent-specific LLM if provided
            agent_type = agent_config.get("type")
            agent = BaseAgent(
                config=agent_config, env=self.environment, model=agent_llm
            )
            agents.append(agent)
            self.logger.debug(
                f"Agent '{agent.agent_id}' of type '{agent_type}' using LLM '{agent_llm}' initialized."
            )
            if isinstance(self.environment, MinecraftEnvironment):
                assert "agent_id" in agent_config and "agent_port" in agent_config
                self.environment.register_agent(
                    agent_config.get("agent_id"), agent_config.get("agent_port")
                )
            self.logger.debug(
                f"Agent '{agent.agent_id}' of type '{agent_type}' initialized."
            )
        return agents

    def _initialize_memory(
        self, memory_config: Dict[str, Any]
    ) -> Union[SharedMemory, BaseMemory]:
        """
        Initialize the shared memory mechanism.

        Args:
            memory_config (dict): Memory configuration.

        Returns:
            BaseMemory: An instance of the memory module.
        """
        memory_type = memory_config.get("type", "SharedMemory")
        memory: Union[BaseMemory, SharedMemory, None] = None
        if memory_type == "SharedMemory":
            memory = SharedMemory()
        else:
            memory = BaseMemory()
        self.logger.debug(f"Memory of type '{memory_type}' initialized.")
        return memory

    def graph_coordinate(self) -> None:
        """
        Graph-based coordination mode.
        """
        try:
            summary_data = {
                "task": self.task,
                "coordination_mode": self.coordinate_mode,
                "iterations": [],
            }
            # Initial assignment: Distribute the overall task to each agent
            self.logger.info("Initial task distribution to all agents.")
            initial_tasks = {
                agent.agent_id: self.task for agent in self.graph.get_all_agents()
            }
            agents_results = []

            # Initialize iteration_data for the initial assignment to match iterative structure
            iteration_data = {
                "iteration": self.current_iteration + 1,
                "task_assignments": {},
                "task_results": [],
                "summary": "",
                "continue_simulation": True,
                "communications": [],
            }
            communications = []
            for agent_id, task in initial_tasks.items():
                try:
                    agent = self.graph.get_agent(agent_id)
                    self.logger.info(f"Assigning initial task to {agent_id}: {task}")
                    # Assign the task to the agent
                    iteration_data_task_assignments = iteration_data.get(
                        "task_assignments"
                    )
                    assert isinstance(iteration_data_task_assignments, dict)
                    iteration_data_task_assignments[agent_id] = task
                    result, communication = agent.act(task)
                    self.logger.info(f"Processing result for agent '{agent.agent_id}'")
                    self.logger.info(f"Communication received: {communication}")
                    if communication:
                        self.logger.info(
                            f"Adding communication to list: {communication}"
                        )
                        communications.append(communication)
                    agents_results.append({agent_id: result})
                    # Record the result
                    task_result = {"agent_id": agent_id, "result": result}
                    iteration_data_task_results = iteration_data.get("task_results")
                    assert isinstance(iteration_data_task_results, list)
                    iteration_data_task_results.append(task_result)
                    self.logger.debug(
                        f"Agent '{agent_id}' completed initial task with result: {result}"
                    )
                except KeyError:
                    self.logger.error(f"Agent '{agent_id}' not found in the graph.")
                except Exception as e:
                    self.logger.error(
                        f"Error while executing initial task for agent '{agent_id}': {e}"
                    )
            iteration_data["communications"] = communications
            # Summarize outputs and update planner for the initial assignment
            summary = self._summarize_results(agents_results)
            self.logger.info(f"Initial Summary:\n{summary}")
            summary = self.planner.summarize_output(
                summary, self.task, self.output_format
            )
            iteration_data["summary"] = summary.content

            # Decide whether to continue or terminate after initial assignment
            if isinstance(self.environment, MinecraftEnvironment):
                try:
                    with open("../data/score.json", "r") as f:
                        block_hit_rate = json.load(f)[-1]["block_hit_rate"]
                except:
                    block_hit_rate = 0.0
                self.logger.info(
                    f"Using a rule-based EnginePlanner. block_hit_rate is {block_hit_rate}"
                )
                continue_simulation = int(block_hit_rate) != 1
            else:
                continue_simulation = self.planner.decide_next_step(agents_results)
            iteration_data["continue_simulation"] = continue_simulation
            if not continue_simulation:
                self.logger.info(
                    "EnginePlanner decided to terminate the simulation after initial assignment."
                )
            else:
                self.planner.update_progress(summary)
                self.current_iteration += 1

            summary_data["iterations"].append(iteration_data)

            # Evaluate communication
            if iteration_data["communications"]:
                iteration_data_communications = iteration_data.get("communications")
                assert isinstance(iteration_data_communications, list)
                communications_str = self._format_communications(iteration_data_communications)
                self.evaluator.evaluate_communication(self.task, communications_str)
            else:
                self.logger.info("No communications to evaluate")
                # Store -1 if communications are empty
                self.evaluator.metrics["communication_score"].append(-1)

            # Evaluate planning
            agent_profiles = self._get_agent_profiles()
            iteration_data_task_assignments = iteration_data.get("task_assignments")
            assert isinstance(iteration_data_task_assignments, dict)
            agent_tasks_str = self._format_agent_tasks(iteration_data_task_assignments)
            iteration_data_task_results = iteration_data.get("task_results")
            assert isinstance(iteration_data_task_results, list)
            results_str = self._format_results(iteration_data_task_results)
            iteration_data_summary = iteration_data.get("summary")
            assert isinstance(iteration_data_summary, str)
            self.evaluator.evaluate_planning(iteration_data_summary, agent_profiles, agent_tasks_str, results_str)
            self.evaluator.evaluate_kpi(self.task, results_str)

            end_on_iter_0 = False
            if not continue_simulation:
                end_on_iter_0 = True

            while self.current_iteration < self.max_iterations and not end_on_iter_0:
                iteration_data = {
                    "iteration": self.current_iteration + 1,
                    "task_assignments": {},
                    "task_results": [],
                    "summary": "",
                    "continue_simulation": True,
                    "communications": [],
                    "total_milestones": 0,
                    "agent_kpis": {},
                }
                self.logger.info(f"Starting iteration {self.current_iteration}")

                current_agents = self.graph.get_all_agents()
                current_tasks = {}
                agents_results = []
                communications = []

                for agent in current_agents:
                    try:
                        # Each agent plans its own task
                        task = agent.plan_task()
                        current_tasks[agent.agent_id] = task
                        iteration_data_task_assignments = iteration_data.get(
                            "task_assignments"
                        )
                        assert isinstance(iteration_data_task_assignments, dict)
                        iteration_data_task_assignments[agent.agent_id] = task
                        self.logger.info(
                            f"Agent '{agent.agent_id}' planned task: {task}"
                        )

                        # Agent acts on the planned task
                        result, communication = agent.act(task)
                        self.logger.info(
                            f"Processing result for agent '{agent.agent_id}'"
                        )
                        self.logger.info(f"Communication received: {communication}")
                        if communication:
                            self.logger.info(
                                f"Adding communication to list: {communication}"
                            )
                            communications.append(communication)
                        agents_results.append({agent.agent_id: result})
                        iteration_data_task_results = iteration_data.get("task_results")
                        assert isinstance(iteration_data_task_results, list)
                        iteration_data_task_results.append({agent.agent_id: result})
                        self.logger.debug(
                            f"Agent '{agent.agent_id}' executed task with result: {result}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error in agent '{agent.agent_id}' during planning or action: {e}"
                        )
                iteration_data["communications"] = communications
                # Summarize outputs and update planner
                summary = self._summarize_results(agents_results)
                self.logger.info(
                    f"Iteration {self.current_iteration} Summary:\n{summary}"
                )
                self.current_iteration += 1
                summary_from_planner = self.planner.summarize_output(
                    summary, self.task, self.output_format
                )
                iteration_data["summary"] = summary_from_planner.content

                # Evaluate communication
                if iteration_data["communications"]:
                    iteration_data_communications = iteration_data.get("communications")
                    assert isinstance(iteration_data_communications, list)
                    communications_str = self._format_communications(iteration_data_communications)
                    self.evaluator.evaluate_communication(self.task, communications_str)
                else:
                    self.logger.info("No communications to evaluate")
                    # Store -1 if communications are empty
                    self.evaluator.metrics["communication_score"].append(-1)

                # Evaluate planning
                agent_profiles = self._get_agent_profiles()
                iteration_data_task_assignments = iteration_data.get("task_assignments")
                assert isinstance(iteration_data_task_assignments, dict)
                agent_tasks_str = self._format_agent_tasks(iteration_data_task_assignments)
                iteration_data_task_results = iteration_data.get("task_results")
                assert isinstance(iteration_data_task_results, list)
                results_str = self._format_results(iteration_data_task_results)
                iteration_data_summary = iteration_data.get("summary")
                assert isinstance(iteration_data_summary, str)
                self.evaluator.evaluate_planning(iteration_data_summary, agent_profiles, agent_tasks_str, results_str)
                self.evaluator.evaluate_kpi(self.task, results_str)
                # Decide whether to continue or terminate
                if isinstance(self.environment, MinecraftEnvironment):
                    try:
                        with open("../data/score.json", "r") as f:
                            block_hit_rate = json.load(f)[-1]["block_hit_rate"]
                    except:
                        block_hit_rate = 0.0
                    self.logger.info(
                        f"Using a rule-based EnginePlanner. block_hit_rate is {block_hit_rate}"
                    )
                    continue_simulation = int(block_hit_rate) != 1
                else:
                    continue_simulation = self.planner.decide_next_step(agents_results)
                
                # Also check communications for judge's final decision (additional safety check)
                if continue_simulation and communications:
                    communications_str = str(communications).lower()
                    # Exclude waiting messages
                    waiting_indicators = [
                        "waiting for both",
                        "not yet received",
                        "before making my final",
                        "i am waiting"
                    ]
                    is_waiting = any(indicator in communications_str for indicator in waiting_indicators)
                    
                    if not is_waiting:
                        judge_final_keywords = [
                            "my final decision is",
                            "i choose",
                            "i select"
                        ]
                        if any(keyword in communications_str for keyword in judge_final_keywords):
                            self.logger.info("Judge's final decision detected in communications. Terminating simulation.")
                            continue_simulation = False
                
                iteration_data["continue_simulation"] = continue_simulation
                summary_data["iterations"].append(iteration_data)
                if not continue_simulation:
                    self.logger.info(
                        "EnginePlanner decided to terminate the simulation."
                    )
                    break

                # # Check if task is completed within the environment
                # if self.environment.is_task_completed():
                #     self.logger.info("Task has been completed successfully.")
                #     break
            # At the end, add the scores to summary_data

            summary_data["planning_scores"] = self.evaluator.metrics["planning_score"]
            summary_data["communication_scores"] = self.evaluator.metrics[
                "communication_score"
            ]
            summary_data["token_usage"] = self._get_totoal_token_usage()
            summary_data["agent_kpis"] = self.evaluator.metrics["agent_kpis"]
            summary_data["total_milestones"] = self.evaluator.metrics[
                "total_milestones"
            ]
            # if self.environment.name == 'Research Environment':
            if isinstance(self.environment, ResearchEnvironment):
                iteration_data_summary = iteration_data.get("summary")
                assert isinstance(iteration_data_summary, str)
                self.evaluator.evaluate_task_research(self.task, iteration_data_summary)
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine graph-based coordination loop completed.")
            elif self.environment.name == "World Simulation Environment":
                self.evaluator.evaluate_task_world(self.task, iteration_data["summary"])
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine graph-based coordination loop completed.")
            elif isinstance(self.environment, MinecraftEnvironment):
                try:
                    with open("../data/score.json", "r") as f:
                        block_hit_rate = json.load(f)[-1]["block_hit_rate"]
                except:
                    block_hit_rate = 0.0
                summary_data["task_evaluation"] = block_hit_rate * 5
            elif self.environment.name == "DB Environment":
                self.evaluator.evaluate_task_db(
                    self.task,
                    iteration_data["summary"],
                    self.config.task["labels"],
                    self.config.task["number_of_labels_pred"],
                    self.config.task["root_causes"],
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine graph-based coordination loop completed.")
            self.logger.info("Engine graph-based coordination loop completed.")

        except Exception:
            self.logger.exception("An error occurred during graph-based coordination.")
            raise
        finally:
            self.evaluator.finalize()
            self.logger.info("Graph-based coordination simulation completed.")
            self._write_to_jsonl(summary_data)

    def star_coordinate(self) -> None:
        """
        Centralized coordination mode.
        """
        try:
            summary_data = {
                "task": self.task,
                "coordination_mode": self.coordinate_mode,
                "iterations": [],
                "final_output": "",
            }
            agents_results: List[Dict[str, Any]] = []
            while self.current_iteration < self.max_iterations:
                iteration_data: Dict[str, Any] = {
                    "iteration": self.current_iteration + 1,
                    "task_assignments": {},
                    "task_results": [],
                    "summary": "",
                    "continue_simulation": True,
                    "total_milestones": 0,
                    "agent_kpis": {},
                }
                self.logger.info(f"Starting iteration {self.current_iteration}")

                # Assign tasks to agents
                assignment = self.planner.assign_tasks(
                    planning_method=self.planning_method
                )
                tasks = assignment.get("tasks", {})
                iteration_data["task_assignments"] = tasks
                self.logger.info(f"Assigned tasks: {tasks}")

                # Assign tasks to agents
                agents_results = []
                communications = []
                for agent_id, task in tasks.items():
                    try:
                        agent = self.graph.get_agent(agent_id)
                        self.logger.info(f"Assigning task to {agent_id}: {task}")
                        result, communication = agent.act(task)
                        agents_results.append({agent_id: result})
                        if communication:
                            communications.append(communication)

                        self.logger.debug(
                            f"Agent '{agent_id}' completed task with result: {result}"
                        )
                    except KeyError:
                        self.logger.error(f"Agent '{agent_id}' not found in the graph.")
                    except Exception as e:
                        self.logger.error(
                            f"Error while executing task for agent '{agent_id}': {e}"
                        )
                iteration_data["task_results"] = agents_results
                iteration_data["communications"] = communications
                # Update progress based on agents' results
                summary = self._summarize_results(agents_results)
                summary_from_planner = self.planner.summarize_output(
                    summary, self.task, self.output_format
                )
                iteration_data["summary"] = summary_from_planner.content
                self.logger.info(summary)
                self.planner.update_progress(summary)
                self.current_iteration += 1

                # Evaluate communication
                if iteration_data["communications"]:
                    communications_str = self._format_communications(
                        iteration_data["communications"]
                    )
                    self.evaluator.evaluate_communication(self.task, communications_str)
                else:
                    # Store -1 if communications are empty
                    self.evaluator.metrics["communication_score"].append(-1)

                # Evaluate planning
                agent_profiles = self._get_agent_profiles()
                agent_tasks_str = self._format_agent_tasks(
                    iteration_data["task_assignments"]
                )
                results_str = self._format_results(iteration_data["task_results"])
                self.evaluator.evaluate_planning(
                    iteration_data["summary"],
                    agent_profiles,
                    agent_tasks_str,
                    results_str,
                )
                self.evaluator.evaluate_kpi(self.task, results_str)

                # Decide whether to continue or terminate
                continue_simulation = self.planner.decide_next_step(agents_results)
                iteration_data["continue_simulation"] = continue_simulation
                summary_data["iterations"].append(iteration_data)
                if not continue_simulation:
                    self.logger.info(
                        "EnginePlanner decided to terminate the simulation."
                    )
                    break

                if self.current_iteration >= self.max_iterations:
                    self.logger.info("Maximum iterations reached.")
                    break
            # At the end, add the scores to summary_data
            summary_data["planning_scores"] = self.evaluator.metrics["planning_score"]
            summary_data["communication_scores"] = self.evaluator.metrics[
                "communication_score"
            ]
            summary_data["token_usage"] = self._get_totoal_token_usage()
            summary_data["agent_kpis"] = self.evaluator.metrics["agent_kpis"]
            summary_data["total_milestones"] = self.evaluator.metrics[
                "total_milestones"
            ]
            if self.environment.name == "Research Environment":
                self.evaluator.evaluate_task_research(
                    self.task, iteration_data["summary"]
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine graph-based coordination loop completed.")
            if self.environment.name == "Coding Environment":
                code = self._read_code_from_file("MARBLE/marble/workspace/solution.py")
                if code:
                    self.evaluator.evaluate_code_quality(
                        task=self.task, code_result=code
                    )
                    summary_data["code_quality"] = self.evaluator.metrics[
                        "code_quality"
                    ]
                    self.logger.info(
                        f"Code quality evaluation results: {self.evaluator.metrics['code_quality']}"
                    )
                self.logger.info("Engine star-based coordination loop completed.")
            elif self.environment.name == "World Simulation Environment":
                self.evaluator.evaluate_task_world(self.task, iteration_data["summary"])
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine star-based coordination loop completed.")
            elif self.environment.name == "DB Environment":
                self.evaluator.evaluate_task_db(
                    self.task,
                    iteration_data["summary"],
                    self.config.task["labels"],
                    self.config.task["number_of_labels_pred"],
                    self.config.task["root_causes"],
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine star-based coordination loop completed.")
            self.logger.info("Engine simulation loop completed.")

        except Exception:
            self.logger.exception("An error occurred during simulation.")
            raise
        finally:
            self.evaluator.finalize()
            self.logger.info("Simulation completed.")
            self._write_to_jsonl(summary_data)

    def chain_coordinate(self) -> None:
        """
        Chain-based coordination mode.
        """
        try:
            self.logger.info("Starting chain-based coordination.")
            summary_data = {
                "task": self.task,
                "coordination_mode": self.coordinate_mode,
                "iterations": [],
            }
            # Start with the initial agent
            current_agent = self._select_initial_agent()
            if not current_agent:
                self.logger.error("No initial agent found for chain.")
                return

            max_chain_length = self.max_iterations * len(
                self.agents
            )  # Or define a separate chain length limit
            chain_length = 0

            task = self.task
            agents_results = []

            while current_agent and chain_length < max_chain_length:
                iteration_data = {
                    "chain_length": chain_length + 1,
                    "current_agent": current_agent.agent_id,
                    "result": None,
                    "continue_simulation": True,
                    "task_assignments": {},
                    "total_milestones": 0,
                    "agent_kpis": {},
                }
                self.logger.info(f"Agent '{current_agent.agent_id}' is executing task.")
                result, communication = current_agent.act(task)
                result_str = f"AgentID: '{current_agent.agent_id}' completed task with result: {result}"
                iteration_data_task_assignments = iteration_data.get("task_assignments")
                assert isinstance(iteration_data_task_assignments, dict)
                iteration_data_task_assignments[current_agent.agent_id] = task
                agents_results.append({current_agent.agent_id: result})
                iteration_data["result"] = result
                self.logger.info(
                    f"Agent '{current_agent.agent_id}' completed task with result: {result}"
                )
                # Get profiles of other agents
                agent_profiles = self.graph.get_agent_profiles_linked(
                    current_agent.agent_id
                )
                # Current agent chooses the next agent
                next_agent_id, plan = current_agent.plan_next_agent(
                    result, agent_profiles
                )
                current_agent_ = current_agent
                try:
                    current_agent = self.graph.get_agent(next_agent_id)
                except Exception:
                    self.logger.error(
                        f"Agent '{next_agent_id}' not found in the graph. keep the same agent."
                    )
                    current_agent = current_agent_
                task = plan
                chain_length += 1
                self.planner.update_progress(result)
                iteration_data["communications"] = communication

                # Evaluate communication
                if iteration_data["communications"]:
                    iteration_data_communications = iteration_data.get("communications")
                    assert isinstance(iteration_data_communications, list)
                    communications_str = self._format_communications(
                        iteration_data_communications
                    )
                    self.evaluator.evaluate_communication(self.task, communications_str)
                else:
                    # Store -1 if communications are empty
                    self.evaluator.metrics["communication_score"].append(-1)

                summary = self._summarize_results(agents_results)
                summary_from_planner = self.planner.summarize_output(
                    summary, self.task, self.output_format
                )
                iteration_data["summary"] = summary_from_planner.content

                # Evaluate planning
                agent_profiles_self = self._get_agent_profiles()
                iteration_data_task_assignments = iteration_data.get("task_assignments")
                assert isinstance(iteration_data_task_assignments, dict)
                agent_tasks_str = self._format_agent_tasks(
                    iteration_data_task_assignments
                )
                iteration_data_summary = iteration_data.get("summary")
                assert isinstance(iteration_data_summary, str)
                self.evaluator.evaluate_planning(
                    iteration_data_summary, agent_profiles_self, agent_tasks_str, result
                )
                self.evaluator.evaluate_kpi(self.task, result_str)

                # Decide whether to continue or terminate
                continue_simulation = self.planner.decide_next_step(
                    [{"root_agent": result}]
                )
                iteration_data["continue_simulation"] = continue_simulation
                summary_data["iterations"].append(iteration_data)
                if not continue_simulation:
                    self.logger.info(
                        "EnginePlanner decided to terminate the simulation."
                    )
                    break
            # Update progress
            summary = self._summarize_results(agents_results)
            self.logger.info(f"Chain execution Summary:\n{summary}")
            self.planner.update_progress(summary)

            # At the end, add the scores to summary_data
            summary_data["planning_scores"] = self.evaluator.metrics["planning_score"]
            summary_data["communication_scores"] = self.evaluator.metrics[
                "communication_score"
            ]
            summary_data["token_usage"] = self._get_totoal_token_usage()
            summary_data["agent_kpis"] = self.evaluator.metrics["agent_kpis"]
            summary_data["total_milestones"] = self.evaluator.metrics[
                "total_milestones"
            ]
            if self.environment.name == "Research Environment":
                self.evaluator.evaluate_task_research(
                    self.task, iteration_data["summary"]
                )
                # summary_data['task_evaluation'] = self.evaluator.metrics["task_evaluation"]
                self.logger.info("Engine chain-based coordination loop completed.")
            elif self.environment.name == "World Simulation Environment":
                self.evaluator.evaluate_task_world(self.task, iteration_data["summary"])
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine chain-based coordination loop completed.")
            elif self.environment.name == "DB Environment":
                self.evaluator.evaluate_task_db(
                    self.task,
                    iteration_data["summary"],
                    self.config.task["labels"],
                    self.config.task["number_of_labels_pred"],
                    self.config.task["root_causes"],
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine chain-based coordination loop completed.")
            self.logger.info("Chain-based coordination simulation completed.")

        except Exception:
            self.logger.exception("An error occurred during chain-based coordination.")
            raise
        finally:
            self.evaluator.finalize()
            self.logger.info("Chain-based coordination simulation completed.")
            summary_data["token_usage"] = self._get_totoal_token_usage()
            self._write_to_jsonl(summary_data)

    def tree_coordinate(self) -> None:
        """
        Tree-based coordination mode.
        """
        try:
            self.logger.info("Starting tree-based coordination.")
            summary_data = {
                "task": self.task,
                "coordination_mode": self.coordinate_mode,
                "iterations": [],
            }

            root_agent = self.graph.get_root_agent()
            if not root_agent:
                self.logger.error("No root agent found in the tree.")
                return
            # Start the coordination from the root agent
            while self.current_iteration < self.max_iterations:
                iteration_data: Dict[str, Any] = {
                    "iteration": self.current_iteration + 1,
                    "root_agent": root_agent.agent_id,
                    "result": None,
                    "continue_simulation": True,
                    "total_milestones": 0,
                    "agent_kpis": {},
                }
                self.current_iteration += 1
                self.logger.info(f"Starting iteration {self.current_iteration}")
                results, communication, tasks = self._execute_agent_task_recursive(
                    root_agent, self.task
                )
                # Update progress
                summary = self._summarize_results(results)
                summary = self.planner.summarize_output(
                    summary, self.task, self.output_format
                )
                iteration_data["summary"] = summary.content
                self.logger.info(
                    f"Iteration {self.current_iteration} Summary:\n{summary}"
                )
                self.planner.update_progress(summary)
                iteration_data["communications"] = communication
                iteration_data["task_assignments"] = tasks
                iteration_data["task_results"] = results
                # Evaluate communication
                if iteration_data["communications"]:
                    communications_str = self._format_communications(
                        iteration_data["communications"]
                    )
                    self.evaluator.evaluate_communication(self.task, communications_str)
                else:
                    # Store -1 if communications are empty
                    self.evaluator.metrics["communication_score"].append(-1)

                # Evaluate planning
                agent_profiles = self._get_agent_profiles()
                agent_tasks_str = self._format_agent_tasks(
                    iteration_data["task_assignments"]
                )
                results_str = self._format_results(iteration_data["task_results"])
                self.evaluator.evaluate_planning(
                    iteration_data["summary"],
                    agent_profiles,
                    agent_tasks_str,
                    results_str,
                )
                self.evaluator.evaluate_kpi(self.task, results_str)

                # Check if task is complete by checking environment's final_translations (for translation tasks)
                # This is more reliable than text parsing
                if hasattr(self.environment, 'final_translations') and isinstance(self.environment.final_translations, dict):
                    if len(self.environment.final_translations) > 0:
                        self.logger.info(
                            f"Final translation detected in environment. Final translations: {self.environment.final_translations}. Terminating simulation."
                        )
                        iteration_data["continue_simulation"] = False
                        summary_data["iterations"].append(iteration_data)
                        break
                
                # Check if the current iteration summary contains a final_translation (even if judge hasn't called judge_decision)
                # Sometimes summaries show completion before the function is actually called
                current_summary = iteration_data.get("summary", "")
                if current_summary and isinstance(current_summary, str) and "translate" in self.task.lower():
                    import re
                    # Look for "final_translation": "..." pattern in summary
                    if '"final_translation"' in current_summary.lower():
                        translation_match = re.search(r'"final_translation"\s*:\s*"([^"]{3,})"', current_summary, re.IGNORECASE)
                        if translation_match:  # Removed iteration >= 3 requirement
                            extracted = translation_match.group(1).strip()
                            # Make sure it's not just waiting/null
                            if (extracted and extracted.lower() not in ["null", "none", ""] and
                                not any(word in extracted.lower() for word in ["waiting", "not yet"])):
                                self.logger.info(
                                    f"Final translation found in iteration {self.current_iteration} summary: '{extracted}'. "
                                    "Terminating simulation."
                                )
                                iteration_data["continue_simulation"] = False
                                summary_data["iterations"].append(iteration_data)
                                break
                
                # Also check the current iteration's summary for final_translation (even if judge hasn't called judge_decision)
                # Sometimes the LLM generates summaries that include final_translation before judge actually calls the function
                current_summary = iteration_data.get("summary", "")
                if current_summary and isinstance(current_summary, str):
                    summary_str_lower = current_summary.lower()
                    # Check if summary mentions final_translation and it's not just waiting
                    if ("final_translation" in summary_str_lower and 
                        not any(indicator in summary_str_lower for indicator in ["waiting", "not yet", "null", "none"])):
                        # Check if there's an actual translation value (not just the key)
                        import re
                        translation_match = re.search(r'"final_translation"\s*:\s*"([^"]{3,})"', current_summary, re.IGNORECASE)
                        if translation_match:
                            extracted_translation = translation_match.group(1).strip()
                            self.logger.info(
                                f"Final translation found in iteration {self.current_iteration} summary: '{extracted_translation}'. "
                                "Terminating simulation even though judge_decision() wasn't called."
                            )
                            iteration_data["continue_simulation"] = False
                            summary_data["iterations"].append(iteration_data)
                            break
                
                # For translation tree mode: Check if both proposer and critic have submitted their final team translations
                # Also check if we can infer completion from the results/summaries even if judge hasn't finalized
                if self.coordinate_mode == "tree" and hasattr(self.environment, 'translations'):
                    translations = self.environment.translations if isinstance(self.environment.translations, dict) else {}
                    proposer_submitted = "proposer" in translations and len(translations.get("proposer", [])) > 0
                    critic_submitted = "critic" in translations and len(translations.get("critic", [])) > 0
                    
                    # Check if both organizers have submitted their team translations
                    if proposer_submitted and critic_submitted:
                        # Both have submitted - check if judge has finalized
                        has_finalized = (hasattr(self.environment, 'final_translations') and 
                                       isinstance(self.environment.final_translations, dict) and
                                       len(self.environment.final_translations) > 0)
                        
                        if not has_finalized:
                            # Both submitted but judge hasn't finalized - check how many iterations we've had
                            # If we're on iteration 4+ and both have submitted, stop the loop
                            # The judge should have finalized by now
                            if self.current_iteration >= 4:
                                self.logger.warning(
                                    f"Iteration {self.current_iteration}: Both proposer and critic have submitted translations, "
                                    "but judge hasn't called judge_decision() to finalize after multiple iterations. "
                                    "Stopping to prevent infinite loop. Final translation will be extracted from available translations."
                                )
                                iteration_data["continue_simulation"] = False
                                summary_data["iterations"].append(iteration_data)
                                break
                    
                    # Also check if we're on iteration 5+ - at this point, we should stop regardless
                    # to prevent infinite loops
                    if self.current_iteration >= 5:
                        self.logger.warning(
                            f"Iteration {self.current_iteration}: Reached maximum safe iteration limit. "
                            "Stopping to prevent infinite loop. Final translation will be extracted from available data."
                        )
                        iteration_data["continue_simulation"] = False
                        summary_data["iterations"].append(iteration_data)
                        break
                
                # Also check if judge_decision was called with "finalize" in the results
                results_str = str(results).lower()
                if "judge_decision" in results_str:
                    if ("decision\": \"finalized" in results_str or "decision\":\"finalized" in results_str or 
                        '"decision": "finalized"' in results_str or "decision: \"finalized" in results_str):
                        self.logger.info(
                            "Judge decision with 'finalize' detected in results. Terminating simulation."
                        )
                        iteration_data["continue_simulation"] = False
                        summary_data["iterations"].append(iteration_data)
                        break
                
                # Decide whether to continue or terminate
                continue_simulation = self.planner.decide_next_step(results)
                iteration_data["continue_simulation"] = continue_simulation
                summary_data["iterations"].append(iteration_data)
                if not continue_simulation:
                    self.logger.info(
                        "EnginePlanner decided to terminate the simulation."
                    )
                    break
            # At the end, add the scores to summary_data
            summary_data["planning_scores"] = self.evaluator.metrics["planning_score"]
            summary_data["communication_scores"] = self.evaluator.metrics[
                "communication_score"
            ]
            summary_data["token_usage"] = self._get_totoal_token_usage()
            summary_data["agent_kpis"] = self.evaluator.metrics["agent_kpis"]
            summary_data["total_milestones"] = self.evaluator.metrics[
                "total_milestones"
            ]
            if self.environment.name == "Research Environment":
                self.evaluator.evaluate_task_research(
                    self.task, iteration_data["summary"]
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine graph-based coordination loop completed.")
            if self.environment.name == "Coding Environment":
                code = self._read_code_from_file("MARBLE/marble/workspace/solution.py")
                if code:
                    self.evaluator.evaluate_code_quality(
                        task=self.task, code_result=code
                    )
                    summary_data["code_quality"] = self.evaluator.metrics[
                        "code_quality"
                    ]
                    self.logger.info(
                        f"Code quality evaluation results: {self.evaluator.metrics['code_quality']}"
                    )
                self.logger.info("Engine tree-based coordination loop completed.")
            elif self.environment.name == "World Simulation Environment":
                self.evaluator.evaluate_task_world(self.task, iteration_data["summary"])
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine tree-based coordination loop completed.")
            elif self.environment.name == "DB Environment":
                self.evaluator.evaluate_task_db(
                    self.task,
                    iteration_data["summary"],
                    self.config.task["labels"],
                    self.config.task["number_of_labels_pred"],
                    self.config.task["root_causes"],
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine tree-based coordination loop completed.")
            self.logger.info("Tree-based coordination simulation completed.")

        except Exception:
            self.logger.exception("An error occurred during tree-based coordination.")
            raise
        finally:
            self.evaluator.finalize()
            self.logger.info("Tree-based coordination simulation completed.")
            self._write_to_jsonl(summary_data)

    def _has_agent_completed_task(self, agent: BaseAgent) -> bool:
        """
        Check if an agent has completed its task based on its role.
        
        Args:
            agent (BaseAgent): The agent to check.
            
        Returns:
            bool: True if agent has completed its task, False otherwise.
        """
        # Check environment for submitted translations (most reliable)
        if hasattr(self.environment, 'translations') and isinstance(self.environment.translations, dict):
            translations = self.environment.translations
            agent_id = agent.agent_id
            
            if agent_id.startswith("translator_"):
                # Translators are complete when they've submitted a translation
                return agent_id in translations and len(translations[agent_id]) > 0
            
            elif agent_id in ["proposer", "critic"]:
                # Organizers are complete when they've submitted their team's final translation
                return agent_id in translations and len(translations[agent_id]) > 0
            
            elif agent_id == "judge":
                # Judge is complete when final_translations dictionary has entries
                if hasattr(self.environment, 'final_translations'):
                    return len(self.environment.final_translations) > 0
        
        # Fallback: check memory for function calls
        memory_str = agent.memory.get_memory_str()
        
        if agent.agent_id.startswith("translator_"):
            # Translators are complete when they've submitted a translation
            return "submit_translation" in memory_str
        
        elif agent.agent_id in ["proposer", "critic"]:
            # Organizers are complete when they've submitted their team's final translation
            return "submit_translation" in memory_str
        
        elif agent.agent_id == "judge":
            # Judge is complete when it has made a final decision
            return "judge_decision" in memory_str
        
        # For other agents, always allow execution
        return False

    def _execute_agent_task_recursive(self, agent: BaseAgent, task: str) -> Any:
        """
        Recursively execute tasks starting from the given agent.

        Args:
            agent (BaseAgent): The agent to execute task.
            task (str): The task to execute.

        Returns:
            Any: The result of the agent's execution.
        """
        # Check if agent has already completed its task
        if self._has_agent_completed_task(agent):
            self.logger.info(f"Agent '{agent.agent_id}' has already completed its task. Skipping re-execution.")
            # Return cached result if available, otherwise return a completion message
            memory_str = agent.memory.get_memory_str()
            cached_result = {
                "agent_id": agent.agent_id,
                "result": f"Agent '{agent.agent_id}' has already completed its task. Previous actions: {memory_str[:200]}"
            }
            return ([cached_result], None, [])
        
        self.logger.info(f"Agent '{agent.agent_id}' is executing task.")
        tasks = []
        print(agent.children)
        if len(agent.children) > 0:
            print("******************start recursive******************")
            # Agent assigns tasks to children
            tasks_for_children = agent.plan_tasks_for_children(task)
            tasks.append(tasks_for_children)
            children_results = []
            communications = []
            for child in agent.children:
                child_task = tasks_for_children.get(child.agent_id, "")
                # If no specific task assigned, use parent's task as fallback
                if not child_task:
                    child_task = task
                    self.logger.warning(
                        f"No task assigned to child '{child.agent_id}', using parent's task as fallback"
                    )
                # Execute child with assigned or fallback task
                (
                    child_result,
                    communication,
                    tasks_,
                ) = self._execute_agent_task_recursive(child, child_task)
                tasks += tasks_
                if communication:
                    communications.append(communication)
                children_results += child_result
            # Agent may also act itself
            results_str = "\n".join(
                json.dumps(result)[:500] for result in children_results
            )

            task_for_father = (
                task
                + "\nHere are the results of the children: "
                + results_str
                + "\nPlease don't repeat the same task and continue to work on the original task. You may also need to communicate with other agents or summarize the results or just continue to work on the original task."
            )
            own_result, communication = agent.act(task_for_father)

            if communication:
                communications.append(communication)
            communications_str = "\n".join(communications) if communications else None
            # # Combine results
            # combined_result = agent.summarize_results(children_results, own_result)
            results = [
                {"agent_id": agent.agent_id, "result": own_result}
            ] + children_results
            return results, communications_str, tasks
        else:
            # Agent directly acts on the task
            result, communication = agent.act(task)
            return (
                [{"agent_id": agent.agent_id, "result": result}],
                communication,
                tasks,
            )

    def _select_initial_agent(self) -> Optional[BaseAgent]:
        """
        Select the initial agent to start the chain.

        Returns:
            Optional[BaseAgent]: The initial agent, or None if not found.
        """
        # Use the first agent in the list as the starting agent
        # This works for any agent names (proposer, critic, judge, etc.)
        if self.agents:
            starting_agent = self.agents[0]
            self.logger.info(f"Selected '{starting_agent.agent_id}' as starting agent for chain.")
            return starting_agent
        else:
            self.logger.error("No agents found for chain coordination.")
            return None

    def start(self) -> None:
        """
        Start the engine to run the simulation.
        """
        self.logger.info("Engine starting simulation.")
        if isinstance(self.environment, MinecraftEnvironment):
            self.environment.launch()
        if self.coordinate_mode == "star":
            self.logger.info("Running in centralized coordination mode.")
            self.star_coordinate()
        elif self.coordinate_mode == "graph":
            self.logger.info("Running in graph-based coordination mode.")
            self.graph_coordinate()
        elif self.coordinate_mode == "chain":
            self.logger.info("Running in chain-based coordination mode.")
            self.chain_coordinate()
        elif self.coordinate_mode == "tree":
            self.logger.info("Running in tree-based coordination mode.")
            self.tree_coordinate()
        else:
            self.logger.error(f"Unsupported coordinate mode: {self.coordinate_mode}")
            raise ValueError(f"Unsupported coordinate mode: {self.coordinate_mode}")
        if isinstance(self.environment, MinecraftEnvironment):
            self.environment.finish()

    def _should_terminate(self) -> bool:
        """
        Determine whether the simulation should terminate.

        Returns:
            bool: True if should terminate, False otherwise.
        """
        # Placeholder for any additional termination conditions
        return False

    def _summarize_results(self, agents_results: List[Dict[str, Any]]) -> str:
        """
        Summarize the agents' results into a string.

        Args:
            agents_results (Dict[str, Any]): The results from all agents.

        Returns:
            str: The summary string.
        """
        summary = "Agents' Results Summary:\n"
        # for agent_id, result in agents_results.items():
        #     summary += f"- {agent_id}: {result}\n"
        for result in agents_results:
            shorten_result = f"- {result}"
            shorten_result = shorten_result[:1000]
            summary += f"{shorten_result}\n"

        self.logger.debug(f"Summarized agents' results:\n{summary}")
        return summary

    def _extract_final_translation(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the judge's final translation decision from summary data.
        Includes translations from proposer and critic, and their presentations to the judge.
        
        Args:
            summary_data (Dict[str, Any]): Full summary data from the simulation.
            
        Returns:
            Dict[str, Any]: Output with final translation, judge decision, agent translations, and presentations.
        """
        final_output = {
            "task": summary_data.get("task", ""),
            "final_translation": None,
            "judge_decision": None,
            "agent_translations": {},  # Translations submitted by proposer and critic
            "presentations_to_judge": {}  # Arguments/presentations given to the judge
        }
        
        # Extract presentations to the judge from communications
        iterations = summary_data.get("iterations", [])
        presentations_to_judge = {"proposer": [], "critic": []}
        
        for iteration in iterations:
            communications = iteration.get("communications") or []
            task_results = iteration.get("task_results") or []
            
            for comm in communications:
                # Communication can be:
                # 1. A string (full_chat_history directly)
                # 2. A dict with "full_chat_history" key
                # 3. A simple string message
                
                chat_history = None
                if isinstance(comm, dict):
                    chat_history = comm.get("full_chat_history", "")
                elif isinstance(comm, str):
                    # Check if it's a serialized chat history (contains "In Session" or "From X to Y:")
                    if "In Session" in comm or ("From " in comm and " to " in comm):
                        chat_history = comm
                    else:
                        # Simple string message - check if it's to judge
                        comm_lower = comm.lower()
                        if "to judge" in comm_lower:
                            if "from proposer to judge" in comm_lower:
                                if ":" in comm:
                                    message = comm.split(":", 1)[-1].strip()
                                    if message and message not in presentations_to_judge["proposer"]:
                                        presentations_to_judge["proposer"].append(message)
                            elif "from critic to judge" in comm_lower:
                                if ":" in comm:
                                    message = comm.split(":", 1)[-1].strip()
                                    if message and message not in presentations_to_judge["critic"]:
                                        presentations_to_judge["critic"].append(message)
                
                # Parse serialized chat history if we have it
                if chat_history and isinstance(chat_history, str) and chat_history.strip():
                    # Parse the serialized chat history
                    # Format: "In Session <id>\nFrom X to Y: message\n..."
                    lines = chat_history.split("\n")
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith("In Session"):
                            continue
                        line_lower = line.lower()
                        
                        # Check for messages TO the judge
                        if "from proposer to judge" in line_lower:
                            # Extract message after ":"
                            if ":" in line:
                                message = line.split(":", 1)[-1].strip()
                                if message and message not in presentations_to_judge["proposer"]:
                                    presentations_to_judge["proposer"].append(message)
                        elif "from critic to judge" in line_lower:
                            if ":" in line:
                                message = line.split(":", 1)[-1].strip()
                                if message and message not in presentations_to_judge["critic"]:
                                    presentations_to_judge["critic"].append(message)
        
        final_output["presentations_to_judge"] = presentations_to_judge
        
        # FIRST: Try to extract from iteration summaries (most reliable source)
        iterations = summary_data.get("iterations", [])
        if iterations:
            for iteration in iterations:
                summary = iteration.get("summary", "")
                # Handle case where summary might be a Message object
                if hasattr(summary, 'content'):
                    summary = summary.content
                if summary and isinstance(summary, str):
                    # Look for "final_translation": "..." pattern in summary JSON
                    if '"final_translation"' in summary.lower() or "'final_translation'" in summary.lower():
                        import re
                        # Try multiple patterns to extract the translation
                        # Pattern 1: Standard JSON format: "final_translation": "text" (allowing newlines with DOTALL)
                        translation_match = re.search(r'["\']final_translation["\']\s*:\s*["\']((?:[^"\'\\]|\\.)*)["\']', summary, re.IGNORECASE | re.DOTALL)
                        if translation_match:
                            extracted = translation_match.group(1).strip()
                            # Unescape any escaped characters
                            extracted = extracted.replace('\\"', '"').replace("\\'", "'").replace('\\n', '\n').replace('\\\\', '\\')
                            # Remove any newlines and extra whitespace
                            extracted = ' '.join(extracted.split())
                            if extracted and self._is_valid_translation(extracted):
                                final_output["final_translation"] = extracted
                                self.logger.info(f"Extracted final translation from iteration summary: '{extracted}'")
                                break  # Found it, no need to check other sources
                        # Pattern 2: Try without strict escaping, simpler pattern
                        if not final_output["final_translation"]:
                            translation_match = re.search(r'["\']final_translation["\']\s*:\s*["\']([^"\']{3,})["\']', summary, re.IGNORECASE)
                            if translation_match:
                                extracted = translation_match.group(1).strip()
                                # Remove any newlines and extra whitespace
                                extracted = ' '.join(extracted.split())
                                if extracted and self._is_valid_translation(extracted):
                                    final_output["final_translation"] = extracted
                                    self.logger.info(f"Extracted final translation from iteration summary (pattern 2): '{extracted}'")
                                    break
                        # Pattern 3: Try to find text between quotes after final_translation (handles multiline)
                        if not final_output["final_translation"]:
                            # Find the position of final_translation
                            match_start = re.search(r'["\']final_translation["\']\s*:\s*["\']', summary, re.IGNORECASE)
                            if match_start:
                                start_pos = match_start.end()
                                # Find the closing quote
                                quote_char = summary[start_pos - 1]  # Get the quote character used
                                end_pos = summary.find(quote_char, start_pos)
                                if end_pos > start_pos:
                                    extracted = summary[start_pos:end_pos].strip()
                                    # Handle escaped quotes
                                    extracted = extracted.replace('\\' + quote_char, quote_char)
                                    # Remove newlines and normalize whitespace
                                    extracted = ' '.join(extracted.split())
                                    if extracted and self._is_valid_translation(extracted):
                                        final_output["final_translation"] = extracted
                                        self.logger.info(f"Extracted final translation from iteration summary (pattern 3): '{extracted}'")
                                        break
        
        # SECOND: Try to extract from iterations task results
        if not final_output["final_translation"] and iterations:
            # Look through all iterations for judge's result
            for iteration in iterations:
                task_results = iteration.get("task_results", [])
                for result_dict in task_results:
                    if "judge" in result_dict:
                        judge_result = result_dict["judge"]
                        if isinstance(judge_result, str):
                            # Exclude waiting messages
                            waiting_indicators = [
                                "waiting for both",
                                "not yet received",
                                "before making my final",
                                "i am waiting",
                                "have not yet received"
                            ]
                            is_waiting = any(indicator in judge_result.lower() for indicator in waiting_indicators)
                            
                            if not is_waiting:
                                # Check if judge made a final decision (positive indicators)
                                decision_keywords = [
                                    "my final decision is",
                                    "i have decided",
                                    "i decided that",
                                    "is the better choice",
                                    "better choice",
                                    "i choose",
                                    "i select",
                                    "final chosen translation",
                                    "decided that",
                                    "after carefully considering"
                                ]
                                if any(keyword in judge_result.lower() for keyword in decision_keywords):
                                    # Store the full judge decision
                                    final_output["judge_decision"] = judge_result
                                    # Also try to extract just the decision part (clean it up)
                                    if "Result from the model:" in judge_result:
                                        # Extract the part after "Result from the model:"
                                        decision_part = judge_result.split("Result from the model:")[-1].strip()
                                        if decision_part:
                                            final_output["judge_decision"] = decision_part
                                
                                # Try to extract the actual translation from the result
                                # Look for English translations (common patterns)
                                # Pattern 1: Text in quotes after "translation" or "final"
                                translation_match = re.search(r'(?:translation|final)[:\s]*["\']([^"\']+)["\']', judge_result, re.IGNORECASE)
                                if translation_match:
                                    final_output["final_translation"] = translation_match.group(1).strip()
                                # Pattern 2: Common English greeting patterns
                                if not final_output["final_translation"]:
                                    translation_match = re.search(r'["\']([^"\']*(?:Hello|Hi|How are you)[^"\']*)["\']', judge_result, re.IGNORECASE)
                                    if translation_match:
                                        final_output["final_translation"] = translation_match.group(1).strip()
                                # Pattern 3: Any quoted English text
                                if not final_output["final_translation"]:
                                    translation_match = re.search(r'["\']([A-Z][^"\']{5,})["\']', judge_result)
                                    if translation_match:
                                        final_output["final_translation"] = translation_match.group(1).strip()
                
                # Also check communications for judge's final decision
                communications = iteration.get("communications") or []
                for comm in communications:
                    if isinstance(comm, str) and "judge" in comm.lower():
                        # Exclude waiting messages
                        waiting_indicators = [
                            "waiting for both",
                            "not yet received",
                            "before making my final",
                            "i am waiting"
                        ]
                        is_waiting = any(indicator in comm.lower() for indicator in waiting_indicators)
                        
                        if not is_waiting:
                            # Check if this communication contains final decision
                            decision_keywords = [
                                "my final decision is",
                                "i have decided",
                                "i decided that",
                                "is the better choice",
                                "better choice",
                                "i choose",
                                "i select",
                                "decided that"
                            ]
                            if any(keyword in comm.lower() for keyword in decision_keywords):
                                final_output["judge_decision"] = comm
                            # Extract translation from communication
                            # Look for English translations in quotes
                            translation_match = re.search(r'["\']([^"\']*(?:Hello|Hi|How are you|translation)[^"\']*)["\']', comm, re.IGNORECASE)
                            if translation_match:
                                final_output["final_translation"] = translation_match.group(1).strip()
                            # Fallback: any quoted text
                            if not final_output["final_translation"]:
                                translation_match = re.search(r'["\']([A-Z][^"\']{5,})["\']', comm)
                                if translation_match:
                                    final_output["final_translation"] = translation_match.group(1).strip()
        
        # If we still don't have a translation, try to extract from summary
        if not final_output["final_translation"]:
            for iteration in iterations:
                summary = iteration.get("summary", "")
                if summary:
                    # Try to extract final_translation from JSON in summary
                    # Use non-greedy matching and validate the result
                    json_match = re.search(r'"final_translation":\s*"((?:[^"\\]|\\.)*)"', summary)
                    if json_match:
                        extracted = json_match.group(1)
                        # Validate: reject obviously malformed translations
                        if self._is_valid_translation(extracted):
                            final_output["final_translation"] = extracted
                            break
        
        # Also check the top-level summary if it exists (from initial task assignment)
        if not final_output["judge_decision"]:
            summary_str = str(summary_data.get("summary", ""))
            if "final_translation" in summary_str.lower() or "judge" in summary_str.lower():
                # Try to extract judge decision from summary JSON
                judge_match = re.search(r'"final_decision":\s*"([^"]+)"', summary_str)
                if judge_match:
                    final_output["judge_decision"] = judge_match.group(1)
                # Try to extract final translation
                if not final_output["final_translation"]:
                    trans_match = re.search(r'"final_translation":\s*"((?:[^"\\]|\\.)*)"', summary_str)
                    if trans_match:
                        extracted = trans_match.group(1)
                        if self._is_valid_translation(extracted):
                            final_output["final_translation"] = extracted
        
        # Last resort: Extract from iteration summaries if we still don't have a translation
        # Look through all iteration summaries for final_translation mentions
        if not final_output["final_translation"]:
            for iteration in iterations:
                summary = iteration.get("summary", "")
                if summary and "final_translation" in summary.lower():
                    # Try multiple patterns to extract
                    patterns = [
                        r'"final_translation":\s*"((?:[^"\\]|\\.)*)"',  # JSON format
                        r'final_translation["\']?\s*[:=]\s*["\']([^"\']+)["\']',  # More flexible
                        r'final translation["\']?\s*[:=]\s*["\']([^"\']+)["\']',  # Case insensitive
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, summary, re.IGNORECASE)
                        if match:
                            extracted = match.group(1).strip()
                            if self._is_valid_translation(extracted):
                                final_output["final_translation"] = extracted
                                self.logger.info(f"Extracted final translation from iteration summary: {extracted}")
                                break
                    if final_output["final_translation"]:
                        break
        
        return final_output
    
    def _is_valid_translation(self, text: str) -> bool:
        """
        Validate that a string is a reasonable translation, not a fragment or malformed text.
        
        Args:
            text: The text to validate
            
        Returns:
            True if the text appears to be a valid translation, False otherwise
        """
        if not text or not isinstance(text, str):
            return False
        
        text = text.strip()
        
        # Reject empty or very short strings
        if len(text) < 3:
            return False
        
        # Reject strings that are clearly fragments/markdown formatting
        invalid_patterns = [
            r'^:\s*$',  # Just a colon
            r'^\*\*.*:\s*$',  # Markdown header ending with colon (like "**MY TRANSLATION:")
            r'^["\']?:\s*\*\*',  # Starts with colon and markdown
            r'^[:\s\n]+$',  # Only whitespace, colons, newlines
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, text):
                return False
        
        # Reject if it's mostly punctuation/formatting
        if len(re.sub(r'[:\s\n*"\']', '', text)) < len(text) * 0.3:
            return False
        
        return True

    def _write_to_jsonl(self, summary_data: Dict[str, Any]) -> None:
        """
        Write summary data to the JSONL file.
        For translation tasks, only writes the judge's final decision.

        Args:
            summary_data (List[Dict[str, Any]]): Summary data to write to the JSONL file.
        """
        file_path = self.config.output.get(
            "file_path", "result/discussion_output.jsonl"
        )
        try:
            # Check if this is a translation task
            task = summary_data.get("task", "").lower()
            is_translation_task = "translate" in task
            
            if is_translation_task:
                # Extract the final translation
                final_output = self._extract_final_translation(summary_data)
                
                # Get translation environment state (includes COMET/BLEURT scores)
                if isinstance(self.environment, TranslationEnvironment):
                    env_state = self.environment.get_state()
                    # Add translation data with scores and rationales
                    final_output["translations"] = env_state.get("translations", {})
                    env_final_translations = env_state.get("final_translations", {})
                    final_output["final_translations"] = env_final_translations
                    
                    # Get input texts
                    input_texts = [
                        {"id": inp.get("id"), "text": inp.get("text")} 
                        for inp in getattr(self.environment, "input_texts", [])
                    ]
                    final_output["input_texts"] = input_texts
                    
                    # Use environment's final_translations as the authoritative source
                    # This is more reliable than text extraction since it comes from judge_decision() function
                    env_translation = None
                    if env_final_translations:
                        # Try to get translation for the current/last input
                        if input_texts and len(input_texts) > 0:
                            current_input_id = input_texts[-1].get("id")
                            if current_input_id and current_input_id in env_final_translations:
                                env_translation = env_final_translations[current_input_id]
                            else:
                                # Fallback: use the first available translation
                                env_translation = list(env_final_translations.values())[0]
                        else:
                            # No input_texts, use first translation
                            env_translation = list(env_final_translations.values())[0]
                    
                    # Prefer environment translation if it's valid
                    if env_translation and self._is_valid_translation(env_translation):
                        final_output["final_translation"] = env_translation
                    elif final_output.get("final_translation"):
                        # Validate extracted translation
                        if not self._is_valid_translation(final_output["final_translation"]):
                            # Invalid extracted translation, clear it
                            final_output["final_translation"] = None
                    
                    # Format agent translations for easier access (proposer and critic)
                    agent_translations = {}
                    translations_dict = env_state.get("translations", {})
                    for agent_id in ["proposer", "critic"]:
                        if agent_id in translations_dict:
                            agent_translations[agent_id] = translations_dict[agent_id]
                    final_output["agent_translations"] = agent_translations
                    
                    # Fallback: If final_translation is still None and we have submitted translations,
                    # use the critic's translation (usually better) or proposer's as last resort
                    if not final_output.get("final_translation") and agent_translations:
                        # Try critic's translation first
                        if "critic" in agent_translations and len(agent_translations["critic"]) > 0:
                            critic_translation = agent_translations["critic"][-1].get("translation")
                            if critic_translation and self._is_valid_translation(critic_translation):
                                final_output["final_translation"] = critic_translation
                                self.logger.warning(f"Using critic's translation as fallback since judge did not finalize")
                        # Fallback to proposer's translation
                        elif "proposer" in agent_translations and len(agent_translations["proposer"]) > 0:
                            proposer_translation = agent_translations["proposer"][-1].get("translation")
                            if proposer_translation and self._is_valid_translation(proposer_translation):
                                final_output["final_translation"] = proposer_translation
                                self.logger.warning(f"Using proposer's translation as fallback since judge did not finalize")
                
                # Also include evaluation metrics
                final_output["evaluation_metrics"] = {
                    "planning_scores": summary_data.get("planning_scores", []),
                    "communication_scores": summary_data.get("communication_scores", []),
                    "token_usage": summary_data.get("token_usage", 0),
                    "agent_kpis": summary_data.get("agent_kpis", {}),
                    "total_milestones": summary_data.get("total_milestones", 0),
                    "task_completion": self.evaluator.metrics.get("task_completion", []),
                    "token_consumption": self.evaluator.metrics.get("token_consumption", [])
                }
                
                # Write as formatted JSON (not JSONL) for translation tasks
                # Read existing data if file exists
                existing_data = []
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                # Try to parse as JSON array first
                                try:
                                    existing_data = json.loads(content)
                                    if not isinstance(existing_data, list):
                                        existing_data = [existing_data]
                                except json.JSONDecodeError:
                                    # If not JSON array, try JSONL (one object per line)
                                    for line in content.split("\n"):
                                        if line.strip():
                                            existing_data.append(json.loads(line))
                    except (json.JSONDecodeError, IOError):
                        existing_data = []
                
                # Add new output
                existing_data.append(final_output)
                
                # Write as formatted JSON array
                with open(file_path, "w", encoding="utf-8") as json_file:
                    json.dump(existing_data, json_file, indent=2, ensure_ascii=False)
                    json_file.flush()
                self.logger.info(f"Final translation with evaluation metrics written to {file_path}")
            else:
                # For non-translation tasks, write full summary
                with open(file_path, "a") as jsonl_file:
                    print(summary_data)
                    jsonl_file.write(json.dumps(summary_data) + "\n")
                    jsonl_file.flush()
                self.logger.info(f"Summary data successfully written to {file_path}")
        except IOError as e:
            self.logger.error(f"Failed to write summary data to {file_path}: {e}")

    def _get_final_ooutput_in_graph(self):
        """
        Get the final output graph.

        Returns:
            Dict[str, Any]: The final output graph.
        """
        return self.graph.get_output_graph()

    def _format_communications(self, communications: List[Any]) -> str:
        """
        Formats the communications list into a string suitable for evaluator input.
        """
        # Assuming each communication is a string or can be converted to string
        return "\n".join(str(c) for c in communications)

    def _get_agent_profiles(self) -> str:
        """
        Retrieves and formats agent profiles into a string.
        """
        agent_profiles = []
        for agent in self.graph.get_all_agents():
            # Assuming agent has attributes agent_id and profile
            agent_profiles.append(
                f"Agent ID: {agent.agent_id}, Profile: {agent.profile}"
            )
        return "\n".join(agent_profiles)

    def _format_agent_tasks(self, agent_tasks: Dict[str, Any]) -> str:
        """
        Formats agent tasks into a string.
        """
        try:
            return "\n".join(
                f"Agent {agent_id}: Task: {task}"
                for agent_id, task in agent_tasks.items()
            )
        except Exception:
            return "\n".join(json.dumps(item) for item in agent_tasks)

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Formats results into a string.
        """
        results_str = []
        for result in results:
            if "agent_id" in result and "result" in result:
                agent_id = result["agent_id"]
                res_content = result["result"]
                results_str.append(f"AgentID: {agent_id}: Result: {res_content}")
            else:
                for agent_id, res_content in result.items():
                    results_str.append(f"Agent {agent_id}: Result: {res_content}")
        return "\n".join(results_str)

    def _get_totoal_token_usage(self) -> int:
        """
        Get the total token usage by all agents.
        """
        return (
            sum(agent.token_usage for agent in self.graph.get_all_agents())
            + self.planner.token_usage
        )
