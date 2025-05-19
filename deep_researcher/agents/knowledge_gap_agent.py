"""
Agent used to evaluate the state of the philosophical research and identify knowledge gaps that still 
need to be addressed.

The Agent takes as input a string in the following format:
===========================================================
ORIGINAL QUERY: <original user query>

HISTORY OF ACTIONS, FINDINGS AND THOUGHTS: <breakdown of activities and findings carried out so far>
===========================================================

The Agent then:
1. Carefully reviews the current research and assesses its completeness in addressing the philosophical query
2. Identifies specific knowledge gaps that still exist and need to be filled
3. Returns a KnowledgeGapOutput object
"""

from pydantic import BaseModel, Field
from typing import List
from .baseclass import ResearchAgent
from ..llm_config import LLMConfig, model_supports_structured_output
from datetime import datetime
from .utils.parse_output import create_type_parser

class KnowledgeGapOutput(BaseModel):
    """Output from the Knowledge Gap Agent"""
    research_complete: bool = Field(description="Whether the philosophical research and findings are complete enough to end the research loop")
    outstanding_gaps: List[str] = Field(description="List of philosophical knowledge gaps that still need to be addressed")


INSTRUCTIONS = f"""
You are a Philosophical Research State Evaluator. Today's date is {datetime.now().strftime("%Y-%m-%d")}.
Your job is to critically analyze the current state of philosophical research, 
identify what knowledge gaps still exist and determine the best next step to take.

You will be given:
1. The original philosophical query and any relevant background context
2. A full history of the tasks, actions, findings and thoughts you've made up until this point in the research process

Your task is to:
1. Carefully review the findings and thoughts, particularly from the latest iteration, and assess their completeness in addressing the philosophical query
2. Determine if the findings are sufficiently complete to end the research loop
3. If not, identify up to 3 knowledge gaps that need to be addressed in sequence

Focus on these aspects when identifying gaps:
- Primary Sources:
  * Missing key philosophical texts
  * Incomplete understanding of original arguments
  * Gaps in historical context
  * Missing interpretations of primary sources

- Philosophical Analysis:
  * Logical gaps in arguments
  * Missing counter-arguments
  * Incomplete critical analysis
  * Unresolved philosophical tensions

- Historical Context:
  * Missing historical background
  * Gaps in philosophical tradition
  * Incomplete understanding of influence
  * Missing connections to other philosophers

- Contemporary Relevance:
  * Missing modern applications
  * Gaps in current interpretations
  * Incomplete understanding of impact
  * Missing contemporary debates

Be specific in the gaps you identify and include relevant philosophical context as this will be passed onto another agent to process without additional context.

Only output JSON and follow the JSON schema below. Do not output anything else. I will be parsing this with Pydantic so output valid JSON only:
{KnowledgeGapOutput.model_json_schema()}
"""

def init_knowledge_gap_agent(config: LLMConfig) -> ResearchAgent:
    selected_model = config.fast_model

    return ResearchAgent(
        name="KnowledgeGapAgent",
        instructions=INSTRUCTIONS,
        model=selected_model,
        output_type=KnowledgeGapOutput if model_supports_structured_output(selected_model) else None,
        output_parser=create_type_parser(KnowledgeGapOutput) if not model_supports_structured_output(selected_model) else None
    )