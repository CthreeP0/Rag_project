from pydantic import BaseModel, Field
from utils import extract_information
from typing import Optional, Dict, Any, List
from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import NodeWithScore
from llama_index.core.llms.llm import LLM

# Define your data models

class CriteriaDecision(BaseModel):
    """The decision made based on a single criteria."""

    scoring: int = Field(description="The scoring in the scale of 1-10 made based on the criteria")
    reasoning: str = Field(description="The reasoning behind the decision")

class ResumeScreenerDecision(BaseModel):
    """The decision made by the resume screener."""
    
    criteria_decisions: List[CriteriaDecision] = Field(description="The decisions made based on the criteria")
    overall_reasoning: str = Field(description="The reasoning behind the overall decision")
    scoring: int = Field(description="The overall scoring in the scale of 1-10 made based on the overall criteria")


QUERY_TEMPLATE = """
[Instruction] You will be provided with details such as the screening criteria, job description and job requirements.
Please act as an impartial judge and evaluate the candidate's {criteria_str} based on the job description and job requirements. For this evaluation, you should primarily consider the following accuracy:
[Accuracy]
Score 1: The candidate's {criteria_str} is completely unrelated to job_description .
Score 3: The candidate's {criteria_str} has minor relevance but does not align with job_description.
Score 5: The candidate's {criteria_str} has moderate relevance but contains inaccuracies to job_description.
Score 7: The candidate's {criteria_str} aligns with job_description. but has minor errors or omissions on either one of them.
Score 10: The candidate's {criteria_str} is completely accurate and aligns very well with job_description.
        

### Job Description
{job_description}

### Screening Criteria
{criteria_str}
"""

# Define your ResumeScreenerPack class
class ResumeScreenerPack(BaseLlamaPack):
    def __init__(self, job_description: str, criteria: str, llm: Optional[LLM] = None) -> None:
        self.reader = extract_information
        llm = llm or OpenAI(model="gpt-3.5-turbo-0125")
        service_context = ServiceContext.from_defaults(llm=llm)
        self.synthesizer = TreeSummarize(output_cls=ResumeScreenerDecision, service_context=service_context)
        self.query = QUERY_TEMPLATE.format(job_description=job_description, criteria_str=criteria)

    def get_modules(self) -> Dict[str, Any]:
        return {"reader": self.reader, "synthesizer": self.synthesizer}

    def run(self, resume_path: str, *args: Any, **kwargs: Any) -> Any:
        docs = self.reader(resume_path)
        output = self.synthesizer.synthesize(query=self.query, nodes=[NodeWithScore(node=doc, score=1.0) for doc in docs])
        return output.response