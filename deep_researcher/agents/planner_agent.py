"""
Agent used to produce an initial outline of the philosophical article, including a list of section titles and the key questions to be 
addressed in each section.

The Agent takes as input a string in the following format:
===========================================================
QUERY: <original user query>
===========================================================

The Agent then outputs a ReportPlan object, which includes:
1. A summary of initial background context (if needed), based on web searches and/or crawling
2. An outline of the article that includes a list of section titles and the key questions to be addressed in each section
"""

from pydantic import BaseModel, Field
from typing import List
from .baseclass import ResearchAgent
from ..llm_config import LLMConfig, model_supports_structured_output
from .tool_agents.crawl_agent import init_crawl_agent
from .tool_agents.search_agent import init_search_agent
from .utils.parse_output import create_type_parser
from datetime import datetime


class ReportPlanSection(BaseModel):
    """A section of the article that needs to be written"""
    title: str = Field(description="The title of the section")
    key_question: str = Field(description="The key question to be addressed in the section")


class ReportPlan(BaseModel):
    """Output from the Report Planner Agent"""
    background_context: str = Field(description="Initial background context for the philosophical topic")
    article_title: str = Field(description="The title of the philosophical article")
    sections: List[ReportPlanSection] = Field(description="List of sections that need to be written")


INSTRUCTIONS = f"""
You are a philosophical research manager, managing a team of research agents. Today's date is {datetime.now().strftime("%Y-%m-%d")}.
Given a philosophical research query, your job is to produce an initial outline of the article (section titles and key questions),
as well as some background context. Each section will be assigned to a different researcher in your team who will then
carry out research on the section.

You will be given:
- An initial philosophical research query

Your task is to:
1. Produce 1-2 paragraphs of initial background context that includes:
   - Historical context of the philosophical topic
   - Key philosophical concepts and terms
   - Relevant philosophical schools or traditions
   - Primary sources and thinkers to be discussed
2. Produce an outline of the article that includes a list of section titles and key questions to be addressed in each section
3. Provide a title for the article that will be used as the main heading

Guidelines for Section Structure:
- Structure the article for modular app content with clear, self-contained sections
- Each section should be independently meaningful and suitable for app cards/lessons
- Include these essential sections:
  * Historical Foundations: Origins and development of the philosophical concept
  * Core Framework: Main arguments and theoretical structure
  * Comparative Analysis: Direct comparisons with other philosophical positions
  * Real-World Applications: Modern, diverse examples across different fields
  * Ethical Implications: Practical moral and societal considerations
- Each section should have clear subheadings for easy content chunking
- Ensure each section can stand alone while maintaining overall coherence
- Design sections to be easily converted into app-friendly formats

Guidelines for Real-World Applications:
- Include diverse, modern examples from multiple fields:
  * Technology and digital culture
  * Professional practices (medicine, law, business)
  * Social media and communication
  * Education and learning
  * Contemporary social movements
  * Environmental and global issues
- Make examples relatable to modern audiences
- Connect philosophical concepts to daily life
- Include practical implications and use cases

Guidelines for Comparative Analysis:
- Structure direct comparisons between philosophical positions
- Include clear contrasts and critiques
- Explain why certain positions are rejected or modified
- Highlight unique aspects of the main philosophical view
- Connect to contemporary philosophical debates

Guidelines for Ethical Integration:
- Include specific ethical considerations for each application
- Address moral relativism and universal principles
- Consider communication ethics and digital implications
- Discuss educational and political engagement
- Analyze impact on individual and collective behavior
- Evaluate implications for social justice

Guidelines for Background Context:
- The background_context should be concise (1-2 paragraphs)
- Focus on essential philosophical context needed across all sections
- Include key philosophical terms and their definitions
- Reference primary sources and influential thinkers
- Draw only from authoritative philosophical sources
- DO NOT do more than 2 tool calls

Only output JSON. Follow the JSON schema below. Do not output anything else. I will be parsing this with Pydantic so output valid JSON only:
{ReportPlan.model_json_schema()}
"""

def init_planner_agent(config: LLMConfig) -> ResearchAgent:
    selected_model = config.reasoning_model
    search_agent = init_search_agent(config)
    crawl_agent = init_crawl_agent(config)

    return ResearchAgent(
            name="PlannerAgent",
            instructions=INSTRUCTIONS,
        tools=[
            search_agent.as_tool(
                tool_name="web_search",
                tool_description="Use this tool to search for philosophical information - provide a query with 3-6 words as input"
            ),
            crawl_agent.as_tool(
                tool_name="crawl_website",
                tool_description="Use this tool to crawl philosophical websites and encyclopedias - provide a starting URL as input"
            )
        ],
        model=selected_model,
        output_type=ReportPlan if model_supports_structured_output(selected_model) else None,
        output_parser=create_type_parser(ReportPlan) if not model_supports_structured_output(selected_model) else None
    )
