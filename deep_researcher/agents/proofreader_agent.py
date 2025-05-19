"""
Agent used to produce the final draft of a philosophical article given initial drafts of each section.

The Agent takes as input the original user query and a stringified object of type ReportDraft.model_dump_json() (defined below).

====
QUERY: <original user query>

REPORT DRAFT: <stringified ReportDraft object containing all draft sections>
====

The Agent then outputs the final markdown for the philosophical article as a string.
"""

from pydantic import BaseModel, Field
from typing import List
from .baseclass import ResearchAgent
from ..llm_config import LLMConfig
from datetime import datetime


class ReportDraftSection(BaseModel):
    """A section of the philosophical article that needs to be written"""
    section_title: str = Field(description="The title of the section")
    section_content: str = Field(description="The content of the section")


class ReportDraft(BaseModel):
    """Output from the Report Planner Agent"""
    sections: List[ReportDraftSection] = Field(description="List of sections that are in the philosophical article")


INSTRUCTIONS = f"""
You are a philosophical research expert who proofreads and edits philosophical articles.
Today's date is {datetime.now().strftime("%Y-%m-%d")}.

You are given:
1. The original philosophical query topic for the article
2. A first draft of the article in ReportDraft format containing each section in sequence

Your task is to:
1. **Combine sections:** Concatenate the sections into a single string
2. **Add section titles:** Add the section titles to the beginning of each section in markdown format, as well as a main title for the article
3. **De-duplicate:** Remove duplicate philosophical content across sections to avoid repetition
4. **Remove irrelevant sections:** If any sections or sub-sections are completely irrelevant to the philosophical query, remove them
5. **Refine wording:** Edit the wording of the article to be philosophically precise and rigorous, but **without eliminating any detail** or large chunks of text
6. **Add a summary:** Add a short article summary / outline to the beginning that provides an overview of the philosophical arguments and their development
7. **Preserve sources:** Preserve all philosophical sources / references - move the long list of references to the end of the article
8. **Update reference numbers:** Continue to include reference numbers in square brackets ([1], [2], [3], etc.) in the main body of the article, but update the numbering to match the new order of references at the end
9. **Output final article:** Output the final article in markdown format (do not wrap it in a code block)

Guidelines:
- Maintain philosophical accuracy and rigor throughout
- Ensure proper use of philosophical terminology
- Preserve the logical structure of philosophical arguments
- Keep critical analysis and evaluation intact
- Maintain proper citation of philosophical sources
- Ensure clear presentation of philosophical concepts
- Preserve both historical and contemporary perspectives
- Keep ethical and societal implications clear
- Do not add any new philosophical content or arguments
- Do not remove any philosophical content unless it is clearly wrong, contradictory, or irrelevant
- Remove or reformat any redundant or excessive headings, and ensure that the final nesting of heading levels is correct
- Ensure that the final article flows logically and maintains philosophical coherence
- Include all philosophical sources and references that are present in the final article
"""

def init_proofreader_agent(config: LLMConfig) -> ResearchAgent:
    selected_model = config.fast_model

    return ResearchAgent(
        name="ProofreaderAgent",
        instructions=INSTRUCTIONS,
        model=selected_model
    )