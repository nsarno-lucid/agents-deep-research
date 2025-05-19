"""
Agent used to determine which specialized agents should be used to address knowledge gaps in philosophical research.

The Agent takes as input a string in the following format:
===========================================================
ORIGINAL QUERY: <original user query>

KNOWLEDGE GAP TO ADDRESS: <knowledge gap that needs to be addressed>

BACKGROUND CONTEXT: <supporting background context related to the original query>

HISTORY OF ACTIONS, FINDINGS AND THOUGHTS: <a log of prior iterations of the research process>
===========================================================

The Agent then:
1. Analyzes the knowledge gap to determine which agents are best suited to address it
2. Returns an AgentSelectionPlan object containing a list of AgentTask objects

The available agents are:
- WebSearchAgent: Search for philosophical information from authoritative sources
- SiteCrawlerAgent: Crawl philosophical websites, encyclopedias, and academic resources
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from ..llm_config import LLMConfig, model_supports_structured_output
from datetime import datetime
from .baseclass import ResearchAgent
from .utils.parse_output import create_type_parser


class AgentTask(BaseModel):
    """A task for a specific agent to address knowledge gaps"""
    gap: Optional[str] = Field(description="The knowledge gap being addressed", default=None)
    agent: str = Field(description="The name of the agent to use")
    query: str = Field(description="The specific query for the agent")
    entity_website: Optional[str] = Field(description="The website of the entity being researched, if known", default=None)


class AgentSelectionPlan(BaseModel):
    """A plan for which agents to use to address knowledge gaps"""
    tasks: List[AgentTask] = Field(description="List of tasks for agents to execute")


INSTRUCTIONS = f"""
You are a Philosophical Research Tool Selector responsible for determining which specialized agents should address knowledge gaps in philosophical research.
Today's date is {datetime.now().strftime("%Y-%m-%d")}.

You will be given:
1. The original philosophical research query
2. A knowledge gap identified in the research
3. A full history of the tasks, actions, findings and thoughts you've made up until this point in the research process

Your task is to decide:
1. Which specialized agents are best suited to address the gap
2. What specific queries should be given to the agents (keep this short - 3-6 words)

Available specialized agents:
- WebSearchAgent: Search for philosophical information from authoritative sources such as:
  * Stanford Encyclopedia of Philosophy
  * Internet Encyclopedia of Philosophy
  * Academic philosophy journals
  * Primary philosophical texts
  * Philosophical dictionaries and reference works
- SiteCrawlerAgent: Crawl specific philosophical websites and resources to gather detailed information about:
  * Primary sources
  * Philosophical arguments
  * Historical context
  * Critical interpretations
  * Contemporary applications

Guidelines:
- Prioritize authoritative philosophical sources
- Focus on primary sources and academic interpretations
- Consider both historical and contemporary perspectives
- Ensure balanced coverage of different philosophical viewpoints
- Include critical analysis and counter-arguments
- Aim to call at most 3 agents at a time in your final output
- You can list the WebSearchAgent multiple times with different queries if needed to cover the full scope of the knowledge gap
- Be specific and concise (3-6 words) with the agent queries - they should target exactly what information is needed
- If you know the website of a specific philosophical resource, always include it in the query
- If a gap doesn't clearly match any agent's capability, default to the WebSearchAgent
- Use the history of actions / tool calls as a guide - try not to repeat yourself if an approach didn't work previously

Only output JSON. Follow the JSON schema below. Do not output anything else. I will be parsing this with Pydantic so output valid JSON only:
{AgentSelectionPlan.model_json_schema()}
"""

def init_tool_selector_agent(config: LLMConfig) -> ResearchAgent:
    selected_model = config.reasoning_model

    return ResearchAgent(
        name="ToolSelectorAgent",
        instructions=INSTRUCTIONS,
        model=selected_model,
        output_type=AgentSelectionPlan if model_supports_structured_output(selected_model) else None,
        output_parser=create_type_parser(AgentSelectionPlan) if not model_supports_structured_output(selected_model) else None
    )
