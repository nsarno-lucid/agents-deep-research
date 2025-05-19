"""
Agent used to write detailed sections of a philosophical article.

The LongWriterAgent takes as input:
1. The original philosophical query
2. The article title
3. A ReportDraft object containing all sections written so far
4. A first draft of the next section to be written

The Agent then:
1. Writes a final draft of the next section with proper philosophical analysis
2. Includes citations and references
3. Returns a LongWriterOutput object
"""
from .baseclass import ResearchAgent, ResearchRunner
from ..llm_config import LLMConfig, model_supports_structured_output
from .utils.parse_output import create_type_parser
from datetime import datetime
from pydantic import BaseModel, Field
from .proofreader_agent import ReportDraft
from typing import List, Tuple, Dict
import re


class LongWriterOutput(BaseModel):
    """Output from the Long Writer Agent"""
    next_section_markdown: str = Field(description="The markdown content for the next section of the philosophical article")
    references: List[str] = Field(description="List of references used in the section")


INSTRUCTIONS = f"""
You are an expert philosophical writer tasked with writing detailed sections of a philosophical article. 
Today's date is {datetime.now().strftime('%Y-%m-%d')}.
You will be provided with:
1. The original philosophical research query
2. The article title
3. A final draft of the article containing all sections written up until this point
4. A first draft of the next section to be written

OBJECTIVE:
1. Write a final draft of the next section that:
   - Maintains philosophical rigor and accuracy
   - Presents arguments clearly and logically
   - Includes proper analysis and interpretation
   - Connects with the broader philosophical context
   - Uses appropriate philosophical terminology
   - Demonstrates deep critical engagement
   - Integrates modern context and applications
   - Is suitable for modular app content
2. Include proper citations and references

Analytical Depth Requirements:
- Go beyond surface-level descriptions to provide deep analysis
- Critically engage with primary sources and interpretations
- Identify and analyze philosophical tensions and contradictions
- Evaluate the strengths and weaknesses of different positions
- Synthesize ideas across different philosophical traditions
- Develop original insights and connections
- Challenge assumptions and explore implications
- Consider counter-arguments and alternative viewpoints

Philosophical Writing Guidelines:
- Present arguments systematically and logically
- Use philosophical terminology accurately
- Include clear definitions of key terms
- Provide examples and analogies for complex concepts
- Present multiple perspectives when relevant
- Maintain critical analysis throughout
- Connect ideas to broader philosophical themes
- Consider both historical and contemporary context

Real-World Applications Guidelines:
- Include diverse, modern examples from multiple fields:
  * Technology and digital culture (social media, AI, digital ethics)
  * Professional practices (medicine, law, business, engineering)
  * Education and learning environments
  * Contemporary social movements
  * Environmental and global issues
  * Personal development and well-being
- Make examples relatable to modern audiences
- Connect philosophical concepts to daily life
- Include practical implications and use cases
- Show how philosophical ideas solve real problems
- Demonstrate relevance to current events and trends

Comparative Analysis Guidelines:
- Structure direct comparisons between philosophical positions
- Include clear contrasts and critiques
- Explain why certain positions are rejected or modified
- Highlight unique aspects of the main philosophical view
- Connect to contemporary philosophical debates
- Show how different positions address similar problems
- Analyze strengths and weaknesses of each approach
- Consider historical and contemporary perspectives

Ethical and Societal Analysis:
- Examine moral and ethical implications
- Consider political and social consequences
- Analyze impact on individual and collective behavior
- Evaluate implications for social justice
- Discuss educational and cultural significance
- Consider environmental and technological implications
- Analyze economic and policy implications
- Evaluate impact on human relationships and society
- Address communication ethics and digital implications
- Consider implications for professional ethics

Citation Guidelines:
- Include numbered citations in square brackets in the text
- List all references at the end of the section
- Follow this example format:

LongWriterOutput(
    next_section_markdown="The concept of the categorical imperative was first introduced by Kant in his Groundwork of the Metaphysics of Morals [1]. This idea has been interpreted in various ways by contemporary philosophers [2].",
    references=["[1] https://example.com/kant-groundwork", "[2] https://example.com/contemporary-interpretations"]
)

Section Structure Guidelines:
- Begin with a clear introduction to the section's topic
- Present key arguments and their development
- Include relevant primary source analysis
- Discuss interpretations and counter-arguments
- Connect to broader philosophical themes
- Conclude with implications and connections
- Ensure deep analysis rather than mere summary
- Develop original insights and connections
- Challenge assumptions and explore implications
- Make content suitable for app-based learning
- Ensure the section can stand alone while maintaining coherence

Content Guidelines:
- You can reformat and reorganize the flow of the content to improve logical structure
- DO NOT remove important philosophical content from the first draft
- Only remove text if it:
  * Is already mentioned earlier in the article
  * Should be covered in a later section
  * Is not relevant to the philosophical inquiry
- Ensure the section heading matches the article outline
- Format the output and references in markdown
- Do not include a title for the references section
- Aim for synthesis rather than mere summary
- Develop original insights and connections
- Challenge assumptions and explore implications
- Make content suitable for app-based learning
- Ensure each section can stand alone while maintaining overall coherence

Only output JSON. Follow the JSON schema below. Do not output anything else. I will be parsing this with Pydantic so output valid JSON only:
{LongWriterOutput.model_json_schema()}
"""

def init_long_writer_agent(config: LLMConfig) -> ResearchAgent:
    selected_model = config.fast_model

    return ResearchAgent(
        name="LongWriterAgent",
        instructions=INSTRUCTIONS,
        model=selected_model,
        output_type=LongWriterOutput if model_supports_structured_output(selected_model) else None,
        output_parser=create_type_parser(LongWriterOutput) if not model_supports_structured_output(selected_model) else None
    )


async def write_next_section(
    long_writer_agent: ResearchAgent,
    original_query: str,
    report_draft: str,
    next_section_title: str,
    next_section_draft: str,
) -> LongWriterOutput:
    """Write the next section of the report"""

    user_message = f"""
    <ORIGINAL QUERY>
    {original_query}
    </ORIGINAL QUERY>

    <CURRENT REPORT DRAFT>
    {report_draft or "No draft yet"}
    </CURRENT REPORT DRAFT>

    <TITLE OF NEXT SECTION TO WRITE>
    {next_section_title}
    </TITLE OF NEXT SECTION TO WRITE>

    <DRAFT OF NEXT SECTION>
    {next_section_draft}
    </DRAFT OF NEXT SECTION>
    """

    result = await ResearchRunner.run(
        long_writer_agent,
        user_message,
    )

    return result.final_output_as(LongWriterOutput)


async def write_report(
    long_writer_agent: ResearchAgent,
    original_query: str,
    report_title: str,
    report_draft: ReportDraft,
) -> str:
    """Write the final report by iteratively writing each section"""

    # Initialize the final draft of the report with the title and table of contents
    final_draft = f"# {report_title}\n\n" + "## Table of Contents\n\n" + "\n".join([f"{i+1}. {section.section_title}" for i, section in enumerate(report_draft.sections)]) + "\n\n"
    all_references = []

    for section in report_draft.sections:
        # Produce the final draft of each section and add it to the report with corresponding references
        next_section_draft = await write_next_section(long_writer_agent, original_query, final_draft, section.section_title, section.section_content)
        section_markdown, all_references = reformat_references(
            next_section_draft.next_section_markdown, 
            next_section_draft.references,
            all_references
        )
        section_markdown = reformat_section_headings(section_markdown)
        final_draft += section_markdown + '\n\n'

    # Add the final references to the end of the report
    final_draft += '## References:\n\n' + '  \n'.join(all_references)
    return final_draft


def reformat_references(
        section_markdown: str, 
        section_references: List[str], 
        all_references: List[str] 
    ) -> Tuple[str, List[str]]:
    """
    This method gracefully handles the re-numbering, de-duplication and re-formatting of references as new sections are added to the report draft.
    It takes as input:
    1. The markdown content of the new section containing inline references in square brackets, e.g. [1], [2]
    2. The list of references for the new section, e.g. ["[1] https://example1.com", "[2] https://example2.com"]
    3. The list of references covering all prior sections of the report

    It returns:
    1. The updated markdown content of the new section with the references re-numbered and de-duplicated, such that they increment from the previous references
    2. The updated list of references for the full report, to include the new section's references
    """
    def convert_ref_list_to_map(ref_list: List[str]) -> Dict[str, str]:
        ref_map = {}
        for ref in ref_list:
            try:
                ref_num = int(ref.split(']')[0].strip('['))
                url = ref.split(']', 1)[1].strip()
                ref_map[url] = ref_num
            except ValueError:
                print(f"Invalid reference format: {ref}")
                continue
        return ref_map

    section_ref_map = convert_ref_list_to_map(section_references)
    report_ref_map = convert_ref_list_to_map(all_references)
    section_to_report_ref_map = {}

    report_urls = set(report_ref_map.keys())
    ref_count = max(report_ref_map.values() or [0])
    for url, section_ref_num in section_ref_map.items():
        if url in report_urls:
            section_to_report_ref_map[section_ref_num] = report_ref_map[url]
        else:
            # If the reference is not in the report, add it to the report
            ref_count += 1
            section_to_report_ref_map[section_ref_num] = ref_count
            all_references.append(f"[{ref_count}] {url}")

    def replace_reference(match):
        # Extract the reference number from the match
        ref_num = int(match.group(1))
        # Look up the new reference number
        mapped_ref_num = section_to_report_ref_map.get(ref_num)
        if mapped_ref_num:
            return f'[{mapped_ref_num}]'
        return ''
    
    # Replace all references in a single pass using a replacement function
    section_markdown = re.sub(r'\[(\d+)\]', replace_reference, section_markdown)

    return section_markdown, all_references


def reformat_section_headings(section_markdown: str) -> str:
    """
    Reformat the headings of a section to be consistent with the report, by rebasing the section's heading to be a level-2 heading

    E.g. this:
    # Big Title
    Some content
    ## Subsection

    Becomes this:
    ## Big Title
    Some content
    ### Subsection
    """
    # If the section is empty, return as-is
    if not section_markdown.strip():
        return section_markdown

    # Find the first heading level
    first_heading_match = re.search(r'^(#+)\s', section_markdown, re.MULTILINE)
    if not first_heading_match:
        return section_markdown

    # Calculate the level adjustment needed
    first_heading_level = len(first_heading_match.group(1))
    level_adjustment = 2 - first_heading_level

    def adjust_heading_level(match):
        hashes = match.group(1)
        content = match.group(2)
        new_level = max(2, len(hashes) + level_adjustment)
        return '#' * new_level + ' ' + content

    # Apply the heading adjustment to all headings in one pass
    return re.sub(r'^(#+)\s(.+)$', adjust_heading_level, section_markdown, flags=re.MULTILINE)
