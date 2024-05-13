from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional,Dict
from datetime import datetime


# Pydantic Class for Candidate
class Candidate(BaseModel):
    """Information about a candidate from his/her resume."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.

    name: Optional[str] = Field(..., description="The name of the candidate")
    phone_number: Optional[str] = Field(
        ..., description="The phone number of the candidate"
    )
    email: Optional[str] = Field(
        ..., description="The email of the candidate"
    )
    local: Optional[str] = Field(
        ..., description="Is the candidate Malaysian(Yes or No)?"
    )
    expected_salary: Optional[str] = Field(
        ..., description="Candidate's expected salary in RM if known. (If the currency is Ringgit Malaysia, assign the numerical value or range values only Eg:'3000-3100'. If in other currency, assign alongside currency)"
    )
    current_location: Optional[List] = Field(
        ..., description="Candidate's current location if known. If the candidate does not mention the country, assign the country based on the state and city. Return it in a python dict format with these three keys, example: {'Country': '', 'State': '', 'City': ''} "
    )
    education_background: Optional[List] = Field(
        ..., description="Every single candidate's education background. (field_of_study, level (always expand to long forms), cgpa (Example: 3.5/4.0), university, start_date, year_of_graduation (Year in 4-digits only, remove month). All in a python dict format."
    )
    professional_certificate: Optional[List] = Field(
        ..., description="Candidate's professional certificates stated in the resume, return each certificate as a string in a python list."
    )
    skill_group: Optional[List] = Field(
        ..., description="Every single candidate's skill groups stated in the resume, return each skills as a string in a python list."
    )
    technology_programs_tool: Optional[List] = Field(
        ..., description="Every single candidate's Technology (Tools, Program, System) related to job title stated in the resume, return each technology as a string in a python list."
    )
    language: Optional[List] = Field(
        ..., description="Languages that is stated in the resume, return each language as a string in a python list."
    )
    previous_job_roles: Optional[List] = Field(
        ..., description="Every single one of the candidate's (job_title, job_company, Industries (strictly classify according to to The International Labour Organization), start_date and end_date (only assign date time format if available. Do not assign duration), job_location, job_duration (return the job_duration in years), return in a python dict format."
    )



# Pydantic Class for Criteria
class Criteria(BaseModel):
    """Hiring criteria based on the job details/ Suggested hiring criteria if not specified."""

    education_background: Optional[str] = Field(
        ..., description="Preferred education backgrounds. If not specified, suggest it based on the job title."
    )
    cgpa: Optional[str] = Field(
        ..., description="Minimum threshold of cgpa for the candidate required for the job. If not specified, suggest it."
    )
    technical_skill: Optional[List] = Field(
        ..., description="Technical skills that are relevant to the job details. If not specified, suggest it based on the job title."
    )
    total_experience_year: Optional[str] = Field(
        ..., description="Minimum/preferred total years of working experience required for the job. If not specified, suggest it based on the applicant category."
    )
    professional_certificate: Optional[List] = Field(
        ..., description="Preferred professional certifications,licenses or accreditations required for the job. If not specified, suggest it based on the job title."
    )
    total_similar_experience_year: Optional[str] = Field(
        ..., description="Minimum/preferred total years of working experience that is related to the job title required for the job. If not specified, suggest it based on the applicant category."
    )
    language: Optional[List] = Field(
        ..., description="Preferred language required for the job. If not specified, suggest it."
    )
    targeted_employer: Optional[List] = Field(
        ..., description="Preferred previous candidate company required for the job. If not specified, suggest it."
    )
    year_of_graduation: Optional[str] = Field(
        ..., description=f"Preferred year of graduation required for the job. If not specified, {datetime.now().year}"
    )
    expected_salary: Optional[str] = Field(
        ..., description="Preferred salary range required for the job. If not specified, suggest the market range in Ringgit Malaysia based on the job title for the applicant category."
    )


# Pydantic Class for Criteria
class Weightage(BaseModel):
    """In the scale of 1-10, weightage assigned to the criteria based on the importance of the criteria in the job details"""

    education_background_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the education_background criteria"
    )
    cgpa_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the cgpa criteria"
    )
    technical_skill_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the technical_skill criteria"
    )
    total_experience_year_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the total_experience_year criteria"
    )
    professional_certificate_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the professional_certificate criteria"
    )
    total_similar_experience_year_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the total_similar_experience_year criteria"
    )
    language_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the language criteria"
    )
    soft_skill_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the soft_skill criteria"
    )
    year_of_graduation_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the year_of_graduation criteria"
    )
    expected_salary_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the expected_salary criteria"
    )


class Job(BaseModel):
    """Data about the job criteria and its weightage."""

    # Creates a model so that we can extract multiple entities.
    criteria: List[Criteria]
    # Creates a model so that we can extract multiple entities.
    weightage: List[Weightage]