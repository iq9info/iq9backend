import os
import json
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from app.models.schemas import ModuleSchema

# Define LangGraph State Schema
class AgentGraphState(TypedDict):
    user_profile: Dict[str, Any]
    topic: str
    sequence_order: int
    is_custom: bool
    module_plan: Dict[str, Any]
    content_items: List[Dict[str, Any]]
    quiz_items: List[Dict[str, Any]]
    final_module: Dict[str, Any]
    validation_passed: bool
    retry_count: int

class Agent1Synthesizer:
    def __init__(self):
        # Initialize LangChain Gemini model interface
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2
        )
        self.graph = self._build_graph()

    # Node 1: Plan Submodule Breakdown
    def plan_submodules(self, state: AgentGraphState) -> Dict[str, Any]:
        profile = state["user_profile"]
        prompt = f"""
        Act as an expert curriculum planner. Design a submodule outline for topic: '{state["topic"]}'.
        Candidate Level: {profile.get("proficiency_level")}, Target Role: {profile.get("target_role")}.
        Return a valid JSON array of submodules covering this topic end-to-end.
        Example output format: [{{"title": "Submodule Name", "sequence_order": 1}}]
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        cleaned_json = response.content.strip().strip("```json").strip("```")
        return {"module_plan": {"submodules": json.loads(cleaned_json)}}

    # Node 2: Generate 50 Learning Content Items
    def generate_content(self, state: AgentGraphState) -> Dict[str, Any]:
        profile = state["user_profile"]
        prompt = f"""
        Act as an AI content generator. Topic: '{state["topic"]}'.
        Submodule structure: {state["module_plan"]}.
        Target proficiency: {profile.get("proficiency_level")}.
        
        Generate 50 Content Items across submodules. 
        Mix content_type strictly between: 'FLASHCARD', 'SHORT_EXPLANATION', 'CODE_EXPLANATION', 'SCENARIO', 'WHAT_IF', and 'SEQUENCING'.
        Return a valid JSON array under the key 'content_items'.
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        cleaned_json = response.content.strip().strip("```json").strip("```")
        parsed = json.loads(cleaned_json)
        items = parsed.get("content_items", parsed) if isinstance(parsed, dict) else parsed
        return {"content_items": items}

    # Node 3: Generate 50 Assessment Items
    def generate_quiz(self, state: AgentGraphState) -> Dict[str, Any]:
        profile = state["user_profile"]
        prompt = f"""
        Act as an AI assessment creator. Topic: '{state["topic"]}'.
        Target proficiency: {profile.get("proficiency_level")}.
        
        Generate 50 Assessment Questions covering all submodules.
        Mix question_type strictly between: 'MCQ', 'TRUE_FALSE', 'PARSONS', 'NODE_ASSEMBLER', 'SCENARIO', 'DEBUGGING', and 'SEQUENCING'.
        Return a valid JSON array under the key 'quiz_items'.
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        cleaned_json = response.content.strip().strip("```json").strip("```")
        parsed = json.loads(cleaned_json)
        items = parsed.get("quiz_items", parsed) if isinstance(parsed, dict) else parsed
        return {"quiz_items": items}

    # Node 4: Validate Pydantic Schema
    def validate_schema(self, state: AgentGraphState) -> Dict[str, Any]:
        try:
            full_module = {
                "module_title": state["topic"],
                "sequence_order": state["sequence_order"],
                "is_custom_topic": state["is_custom"],
                "submodules": state["content_items"],
                "assessment_questions": state["quiz_items"]
            }
            validated = ModuleSchema.model_validate(full_module)
            return {"final_module": validated.model_dump(), "validation_passed": True}
        except Exception:
            return {
                "validation_passed": False, 
                "retry_count": state.get("retry_count", 0) + 1
            }

    # Conditional Routing Logic (Self-Healing Loop)
    def route_validation(self, state: AgentGraphState) -> str:
        if state["validation_passed"]:
            return "end"
        elif state["retry_count"] >= 3:
            return "fallback"
        return "retry_content"

    # Assemble State Graph Workflow
    def _build_graph(self):
        workflow = StateGraph(AgentGraphState)

        # Add Nodes
        workflow.add_node("planner", self.plan_submodules)
        workflow.add_node("content_builder", self.generate_content)
        workflow.add_node("quiz_builder", self.generate_quiz)
        workflow.add_node("validator", self.validate_schema)

        # Add Direct Edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "content_builder")
        workflow.add_edge("content_builder", "quiz_builder")
        workflow.add_edge("quiz_builder", "validator")

        # Add Conditional Edges
        workflow.add_conditional_edges(
            "validator",
            self.route_validation,
            {
                "end": END,
                "retry_content": "content_builder",
                "fallback": END
            }
        )

        return workflow.compile()

    # Public method matching previous signature
    def generate_module_content(self, topic: str, user_profile: dict, sequence_order: int, is_custom: bool = False) -> ModuleSchema:
        initial_state: AgentGraphState = {
            "user_profile": user_profile,
            "topic": topic,
            "sequence_order": sequence_order,
            "is_custom": is_custom,
            "module_plan": {},
            "content_items": [],
            "quiz_items": [],
            "final_module": {},
            "validation_passed": False,
            "retry_count": 0
        }
        
        result = self.graph.invoke(initial_state)
        
        if result.get("validation_passed"):
            return ModuleSchema.model_validate(result["final_module"])
        else:
            raise ValueError(f"LangGraph failed schema validation for topic: {topic}")