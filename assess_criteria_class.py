from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
import time
import ast
import os
from datetime import datetime
from openai import OpenAI


class JobParser:
    def __init__(self, job_title:str, job_description:str,job_requirement:str):
        self.job_title = job_title.lower()
        self.job_description = job_description
        self.job_requirement = job_requirement

    def extract_additional_skills(self):
        # Summarize job description
        client = OpenAI()
        response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125", #3.5 turbo
                messages=[
                    {"role": "system", "content": f"""Assume yourself as a hiring manager, you will be provided the job description and job requirement for {self.job_title}.
                     1. Extract the skills related to {self.job_title} from the text.
                     2. Output only the skills without any addiitonal reasonings.
                     3. The output result should strictly follows the python list format.
                     Example: ['Python','SQL','Hadoop']"""},
                    {"role": "user", "content": f"""[Start of Job Description]
                     {self.job_description}
                     [End of Job Description]
                     
                    [Start of Job Requirement]
                     {self.job_requirement}
                     [End of Job Requirement]
                     """}
                ],
            temperature=0,
            max_tokens=600,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0)
        jobdescription = response.choices[0].message.content
        
        jd_skills = ast.literal_eval(jobdescription)
        print("printing skills from jobdescription",jd_skills)
        
        self.jd_skills = jd_skills

        return self
    
class ResumeParser:
    def __init__(self, job_title,job_description):
        self.job_title = job_title
        self.job_description = job_description
        self.current_date = datetime.now()
        self.targEmp_industries_included = [] # from xlsx for 'included' ONLY

    def evaluate_education_background(self, data_dict,input,weightage):
        max_retries = 5
        retry_count = 0
        try:
            edu_prompt_system = f"""[Instruction] You will be provided with details such as the preferred field of study, job_title, and the candidate's field of study.
            Please act as an impartial judge and evaluate the candidate's field of study based on the job title and preferred education background. For this evaluation, you should primarily consider the following accuracy:
            [Accuracy]
            Score 1: The candidate's field of study is completely unrelated to {input} and the job title stated.
            Score 3: The candidate's field of study has minor relevance but does not align with {input} and the job title stated.
            Score 5: The candidate's field of study has moderate relevance but contains inaccuracies to {input} and the job title stated.
            Score 7: The candidate's field of study aligns with {input} and the job title stated but has minor errors or omissions on either one of them.
            Score 10: The candidate's field of study is completely accurate and aligns very well with {input} and the job title stated.
            
            [Rules]
            1. If the candidate has several education background, you should always consider the most related to {input} and the job title only.
            2. You should always ignore those that are unrelated to {input} and the job title and make sure they do not affect the total scoring.
            3. You should only assess the candidate's Field of Study and it's level. Ignore any other criterias.

            [Steps]
            Step 1 : Start the evaluation by giving reasons, Be as objective as possible.
            Step 2 : You must rate the candidate on a scale of 1 to 10 by strictly following this format: "[[rating]]", 
            for example:
            "Education Background Rating: [[6]].

            [Question]
            How will you rate the candidate's education background based on the provided job title with preferred education background?
            """

            edu_prompt_user = f"""
            Preferred Field of Study: {input}
            
            Job Title: {self.job_title}

            [The Start of Candidate's Education Background]
            {data_dict['Education Background']}
            [The End of Candidate's Education Background]
            """
            client = ChatOpenAI()
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": edu_prompt_system},
                    {"role": "user", "content": edu_prompt_user}
                ],
                model="gpt-35-turbo-0125",
                temperature=0.3,
                n=3,
            )
            print("Response from edu",response)
        except openai.RateLimitError as e:
            print(f"OpenAI rate limit exceeded. Pausing for one minute before resuming... (From RateLimitError)")
            print(e)
            time.sleep(30)
            retry_count += 1

            if retry_count >= max_retries:
                print("Exceeded maximum retries for evaluating education background.... (From RateLimitError)")
                return response
        
        
        # Extrdact the number using regex
        def extract_gpt_response_rating(response):
            ratings = []
            pattern = r'\[\[([\d]+)\]\]'

            for i in range(len(response.choices)):
                match = re.search(pattern, response.choices[i].message.content)
                if match:
                    rating = int(match.group(1))
                    ratings.append(rating)
                else:
                    # ratings = 0
                    ratings.append(0)
            return ratings
        # takes in list of ratings from gpt rating response 
        def calculate_average_rating(ratings):
            if not ratings:
                return 0
            return round(sum(ratings) / len(ratings))

        def calculate_weighted_score(average_rating, weightage):
            if average_rating is None:
                return 0
            return round(average_rating/10*weightage)
                
        edu_rating = extract_gpt_response_rating(response)
        average_rating = calculate_average_rating(edu_rating)
        edu_weighted_score = calculate_weighted_score(average_rating,weightage)
        print(f"Candidate: {data_dict['Name']}\t\t1. EDU Score:{edu_weighted_score}/{weightage}\t C: refer data_dict E: {input}\t ")
        
        pos = list(data_dict.keys()).index('Education Background')
        items = list(data_dict.items())
        items.insert(pos+1, ('Education Background Score', edu_weighted_score))
        data_dict = dict(items)
        return data_dict

    def evaluate_cgpa(self,criterion_key, data_dict,input_cgpa, input_edu_b, weightage):
        out_weighted_cgpa_score = 0.0


        def get_normalize_cgpa(cgpa_str,standard_scale = 4.0):
            # Regex pattern to match CGPA values and their max scales
            pattern = r'(\d+(?:\.\d+)?)(?:/(\d+(?:\.\d+)?))?'

            # Searching for the pattern in the text
            match = re.search(pattern, cgpa_str)
            if match:
                cgpa = float(match.group(1))
                max_cgpa = float(match.group(2)) if match.group(2) else standard_scale

                # Normalize CGPA to the standard scale
                normalized_cgpa = (cgpa / max_cgpa) * standard_scale
                print (f"""normalised cgpa:  {normalized_cgpa}, raw cgpa extracted: {cgpa_str}""")
                return normalized_cgpa
            else: # if N/A in resume, cpga -> 0.0 
                print ("normalised cgpa:  CPGA not found. Default CGPA = 0.0/4.0")
                return float("0")


        if 'Education Background' not in data_dict or not isinstance(data_dict['Education Background'], list):
            data_dict[f'{criterion_key}'] = "No CGPA detected. Reason: Education Background not detected"
            data_dict[f'{criterion_key} Score'] = 0.4 * weightage
        else: 
            pos = list(data_dict.keys()).index("Education Background Score") # assumption: we can only get cgpa Education Background is selected, add to the right 
            items = list(data_dict.items())
            items.insert(pos+1, (f'{criterion_key}', ""))
            items.insert(pos+2, (f'{criterion_key} Score', 0))
            data_dict = dict(items)

            c_cgpa = 0 #total 
            edu_cgpa = 0 #each edu 
            count = 0

            print ("CGPA method 2: Getting latest available cgpa")
            data_dict['Education Background'].sort(key=lambda x: x['Year of Graduation'], reverse=True)
            for education_record in data_dict['Education Background']:
                if type(education_record) is dict:
                    # data_dict['Education Background'].sort(key=lambda x: x['Year of Graduation'], reverse=True)
                    if education_record["CGPA"]  != "N/A" :
                        c_cgpa = get_normalize_cgpa(education_record['CGPA'])
                        break

            if count > 1:
                c_cgpa /= count  # Calculate the average normalised CGPA
            if float(c_cgpa) >= float(input_cgpa):
                out_weighted_cgpa_score = 1.0 * weightage
            else:
                out_weighted_cgpa_score = 0.4 * weightage 
            print(f"Candidate: {data_dict['Name']}\t\t 2. CGPA Score:{out_weighted_cgpa_score}/{weightage}\t C CGPA(normalised): {c_cgpa} VS E: {input_cgpa}, {input_edu_b} \t ")

            data_dict[f'{criterion_key}'] = c_cgpa
            data_dict[f'{criterion_key} Score'] = out_weighted_cgpa_score
        return data_dict

    def evaluate_skill_groups(self,data_dict,input,weightage):
        
        JD = JobParser(self.pdf_dir, self.job_title)
        JD_skills = JD.extract_additional_skills()
        result_list = [skill.strip().lower() for skill in input.split(",")]
        data_dict_lower = [x.lower() for x in data_dict['Skill Groups']]
        # Convert all strings in the list to lowercase
        jd_skills_lower = [x.lower() for x in JD_skills.jd_skills]

        if not data_dict_lower or (len(data_dict_lower) == 1 and data_dict_lower[0] == 'N/A'):  # If the list is empty or contains only 'N/A'
            pos = list(data_dict.keys()).index('Skill Groups')
            items = list(data_dict.items())
            items.insert(pos+1, ('Skill Groups Score', 0))
            data_dict = dict(items)
            return data_dict
                
        #Define embeddings model
        embeddings_model = AzureOpenAIEmbeddings(azure_deployment='ada002')

        #Embeds both list
        embedding1 = embeddings_model.embed_documents(data_dict_lower) #candidate skill groups
        embedding2 = embeddings_model.embed_documents(jd_skills_lower+result_list) #required skill groups

        #Calculate the cosine similarity score from embeddings
        similarity_test = cosine_similarity(embedding1,embedding2)

        def similarity_range_score(similarity_scores):
            categorical_scores = []

            for score in similarity_scores:
                if score >= 0.88:
                    categorical_scores.append(1.0)
                elif score >= 0.85:
                    categorical_scores.append(0.5)
                elif score >= 0.8:
                    categorical_scores.append(0.3)
                else:
                    categorical_scores.append(0.0)
            print(categorical_scores)

            return categorical_scores

            
        res = round(np.mean(similarity_range_score(similarity_test.max(axis=0)))*weightage,2)
        
        
        pos = list(data_dict.keys()).index('Skill Groups')
        items = list(data_dict.items())
        items.insert(pos+1, ('Skill Groups Score', res))
        data_dict = dict(items)
        
        print(f"Candidate: {data_dict['Name']}\t\t3. SkillGroup Score:{res}/{weightage}\tC similairty score: {res} E: {input} \t ")
            
        return data_dict

    def evaluate_total_working_exp_years(self, data_dict, input_string, weightage):
        
        criterion_key,c_total_yr_exp, out_weighted_score = "Years of Total Work Experience", 0.0, 0.0

        pos = list(data_dict.keys()).index("Previous job roles")
        items = list(data_dict.items())
        items.insert(pos+1, (f'{criterion_key}', ""))
        items.insert(pos+2, (f'{criterion_key} Score', ""))
        data_dict = dict(items)
        
        def parse_date(date_str):
            # Handle other values gracefully
            if date_str.lower() in ["n/a", "none"]:
                return None
            elif date_str.lower() in [ "present", "current", "now"]: 
                return datetime.now()     
            # Expanded corrections for non-standard month abbreviations to standard ones
            corrections = {
                "Jan": "Jan", "Feb": "Feb", "Mar": "Mar", "Apr": "Apr",
                "May": "May", "Jun": "Jun", "Jul": "Jul", "Aug": "Aug",
                "Sep": "Sep", "Sept": "Sep",  # Both 'Sep' and 'Sept' to 'Sep'
                "Oct": "Oct", "Nov": "Nov", "Dec": "Dec",
                "Mac": "Mar",  # Non-standard, common in some regions
                # Add additional non-standard abbreviations as needed
            }
            # Replace non-standard abbreviations with their standard equivalents
            for incorrect, correct in corrections.items():
                if incorrect in date_str:
                    date_str = date_str.replace(incorrect, correct)

            # Extensive list of date formats to try parsing the date strings
            date_formats = [
                "%Y",
                "%B %Y",  # Full month name and year
                "%m %Y",
                "%d %m %Y",
                "%d-%m-%Y",
                "%Y-%m-%d",
                "%b %Y",  # Abbreviated month name and year
                "%Y-%m",
                "%m-%Y",
                "%d %B %Y",
                "%d %b %Y",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%m/%d/%Y",
                "%m/%d/%y",
                "%d/%m/%Y",
                "%d/%m/%y",
            ]

            for fmt in date_formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue  # If current format fails, try the next

            # If all formats fail, print an error and return None
            print(f"Error parsing date '{date_str}': does not match expected formats.")
            return None
        # def parse_date(date_str):
        #     # Handle other values gracefully
        #     if date_str.lower() in ["n/a", "none"]:
        #         return None
        #     elif date_str.lower() in [ "present", "current", "now"]: 
        #         return datetime.now()     
            
        #     # Define multiple formats to try parsing the date strings
        #     date_formats = ["%Y",
        #                     "%B %Y",  # Full month name and year
        #                     "%m %Y",
        #                     "%d %m %Y",  # 
        #                     "%Y-%m-%d",  # Year-month-day
        #                     "%b %Y",  # Abbreviated month name and year
        #                     "%Y-%m-%d %H:%M:%S.%f"]  # Full datetime format

        #     for fmt in date_formats:
        #         try:
        #             return datetime.strptime(date_str, fmt)
        #         except ValueError:
        #             continue  # If current format fails, try the next

        #     # If all formats fail, print an error and return None
        #     print(f"Error parsing date '{date_str}': does not match expected formats.")
        #     return None


        # def parse_date(date_str, is_start_date=True):
        #     # Handle non-date values gracefully
        #     if date_str.lower() in ["n/a", "none"]:
        #         return None
        #     elif date_str.lower() in [ "present", "current", "now"]: 
        #         return datetime.now()
        #     try:
        #         # Assuming the format for start date is "Mon YYYY" and end date is "YYYY-MM-DD HH:MM:SS.SSSSSS"
        #         if is_start_date:
        #             return datetime.strptime(date_str, "%b %Y")
        #         else:
        #             return datetime.strptime(date_str.split(" ")[0], "%Y-%m-%d")
        #     except ValueError as e:
        #         print(f"Error parsing date '{date_str}': {e}")
        #         return None
            
        # def parse_date(date_str, is_start_date=True):
        #     if date_str.lower() == 'present' or date_str.lower() == 'current':
        #         return datetime.now()
        #     else:
        #         try:
        #             return parser.parse(date_str)
        #         except ValueError:
        #             print ("parse date error")
        #             # Date format not recognized, return today's date for start date,
        #             # or the earliest possible date for end date
        #             if is_start_date:
        #                 # return datetime.now()
        #                 return date_str
        #             else:
        #                 # return datetime.min
        #                 return date_str
        # Calculate the duration in years
        def calculate_duration(start_date, end_date):
            if start_date is None or end_date is None:
                return 0
            # if year given only, assume 6 months 
            if start_date.year == end_date.year: 
                return 0.5  #year
            else: 
                duration = (end_date - start_date).days / 365.25  # Adjusting for leap years
            return duration
        

        # def manual_cal_total_exp (): 
        #     total_duration = 0

        #     # 1. Processing each job role + manually adding up durations to decimal points 
        #     print ("METHOD: Manual calculation on total years of exp fields added:")
        #     # Ensure data_dict['Previous job roles'] is a list of dictionaries
        #     if isinstance(data_dict['Previous job roles'], list):
        #         # Process each job role
        #         for role in data_dict['Previous job roles']:
        #             start_date = parse_date(role.get('Start Date', ''))
        #             end_date = parse_date(role.get('End Date', ''), is_start_date=False)
        #             if str(start_date).lower() != "n/a" and end_date != "n/a": 
        #                 duration = calculate_duration(start_date, end_date)
        #                 total_duration += duration
        #                 print(f"Job Title: {role.get('Job Title', '')}, Duration: {duration:.2f} years")
        #                 role['Job Duration (Years)'] = round(duration, 2)  # Append duration for each role
        #             else: 
        #                 print ("Manual total year of experience: start or end date format invalid")

        #     else:
        #         print("Previous job roles data is not in the expected format.")
        #     print (f"Manual total yr: {total_duration}")
        #     return total_duration

        def manual_cal_total_exp():
            total_duration = 0
            print("METHOD: Manual calculation on total years of exp fields added:")
            if isinstance(data_dict.get('Previous job roles', None), list):
                for role in data_dict['Previous job roles']:
                    start_date = parse_date(role.get('Start Date', ''))
                    end_date = parse_date(role.get('End Date', ''))
                    if start_date and end_date:  # Ensure both dates are valid
                        duration = calculate_duration(start_date, end_date)
                        total_duration += duration
                        print(f"Job Title: {role.get('Job Title', 'Unknown')}, Duration: {duration:.2f} years")
                        role['Job Duration (Years)'] = round(duration, 2)  # Append calculated duration for each role
                    else:
                        print("Start or end date format invalid or missing for one of the roles.")
            else:
                print("Previous job roles data is not in the expected format.")
            print(f"Manual total yr: {total_duration:.2f}")
            return total_duration
        
        def gpt_calc_total_exp():
            # Check if 'Previous job roles' exists and is a list
            if not isinstance(data_dict.get('Previous job roles', None), list):
                return "Data structure is incorrect or 'Previous job roles' is missing."
            
            total_duration = 0
            for role in data_dict['Previous job roles']:
                try:
                    # Attempt to convert job duration to float and add to total
                    duration_str = role.get("Job Duration (Years)", "0")  # Default to "0" if not found
                    duration = float(duration_str)
                    total_duration += duration
                except ValueError:
                    # Handle case where conversion to float fails
                    print(f"Error converting job duration to float for role: {role.get('Job Title')}. Skipping this entry.")
                    continue  # Skip this entry and continue with the next
            print (f"gpt4 total yr: {total_duration}")
                    
            return round(total_duration, 2)

        # Manual: Total duration
        total_duration_manual = manual_cal_total_exp()
        total_experience_gpt4 = gpt_calc_total_exp()
        # print(f"Total yr_exp in years: {total_experience_gpt4:.2f}") #cause ValueError: Unknown format code 'f' for object of type 'str'
        # Use parse_range to get the lower and upper limits and condition
        in_threshold_lower_limit, in_threshold_upper_limit, condition = self.parse_range(input_string)
        try:
            c_total_yr_exp = float(total_experience_gpt4)
            if c_total_yr_exp < in_threshold_lower_limit:
                out_weighted_score = 0  # does not meet requirement
            elif in_threshold_lower_limit <= c_total_yr_exp <= in_threshold_upper_limit:
                out_weighted_score = 1.0 * weightage  # within range ir equal 
            elif c_total_yr_exp > in_threshold_upper_limit:
                out_weighted_score = 0.5 * weightage  # overqualified
            else:
                out_weighted_score = 0
            print(f"Candidate: {data_dict['Name']}\t\t4.Total years of experience Score:{out_weighted_score}/ {weightage}\t C:{c_total_yr_exp}, Required years: {input_string}\n ")
        except ValueError:
            # Handle the case where conversion to float fails
            out_weighted_score = 0  

        data_dict[f'{criterion_key}'] = total_experience_gpt4
        data_dict[f'{criterion_key} Score'] = out_weighted_score
        
        print("data_dict after twe",data_dict)
        return data_dict

    def evaluate_technology(self,data_dict,input,weightage):
        print("tech now",data_dict)
        result_list = [skill.strip().lower() for skill in input.split(",")]
        #Define embeddings model
        embeddings_model = AzureOpenAIEmbeddings(azure_deployment='ada002')
        data_dict_lower = [x.lower() if isinstance(x, str) else x for x in data_dict['Technology (Tools, Program, System)']]

        if not data_dict_lower or (len(data_dict_lower) == 1 and data_dict_lower[0] == 'N/A'):  # If the list is empty or contains only 'N/A'
            pos = list(data_dict.keys()).index('Technology (Tools, Program, System)')
            items = list(data_dict.items())
            items.insert(pos+1, ('Technology (Tools, Program, System) Score', 0))
            data_dict = dict(items)
            print(data_dict)
            return data_dict

        #Embeds both list
        embedding1 = embeddings_model.embed_documents(data_dict_lower) #candidate skill groups
        embedding2 = embeddings_model.embed_documents(result_list) #required skill groups

        #Calculate the cosine similarity score from embeddings
        similarity_test = cosine_similarity(embedding1,embedding2)

        def similarity_range_score(similarity_scores):
            categorical_scores = []

            for score in similarity_scores:
                if score >= 0.88:
                    categorical_scores.append(1.0)
                elif score >= 0.85:
                    categorical_scores.append(0.5)
                elif score >= 0.8:
                    categorical_scores.append(0.3)
                else:
                    categorical_scores.append(0.0)
            print(categorical_scores)

            return categorical_scores
            
        res = round(np.mean(similarity_range_score(similarity_test.max(axis=0)))*weightage,2)
        pos = list(data_dict.keys()).index('Technology (Tools, Program, System)')
        items = list(data_dict.items())
        items.insert(pos+1, ('Technology (Tools, Program, System) Score', res))
        data_dict = dict(items)
        
        print(f"Candidate: {data_dict['Name']}\t\t3. Tech Score:{res}/{weightage}\tC similairty score: {res} E: {input} \t ")

        return data_dict


    def evaluate_year_exp_role(self, data_dict, input, weightage, index):

        def extract_yoer_similar(data_dict):
            max_retries = 5
            retry_count = 0 
            try:
                yoer_prompt_system = f"""[Instruction] 
                You will be provided with details such as the candidate's previous job roles. Please act as a hiring manager with 20 years experience to evaluate the candidate's previous job roles.
                1. Identify job roles that are similar to {self.job_title}. You should also consider roles that are related to {self.job_title}.
                2. Output all of the duration of the related job roles into a python list.
                3. The output format should strictly follow the format in the example provided.
                Example of the output: Total duration: [[2,3,4]]

                [Question]
                What are the job durations for the job roles that are related to {self.job_title} in the candidate's previous job experience?
                """

                yoer_prompt_user = f"""
                Candidate's Previous Job Roles: {data_dict["Previous job roles"]}
                """
                client = AzureOpenAI()
                response = client.chat.completions.create(
                    model="gpt-35-turbo-16k", # 3.5 turbo
                    messages=[
                        {"role": "system", "content": yoer_prompt_system},
                        {"role": "user", "content": yoer_prompt_user}
                    ],
                    temperature=0.3,
                )
                print("Response from yoer",response)
                return response.choices[0].message.content
            except openai.RateLimitError as e:
                print(f"OpenAI rate limit exceeded. Pausing for one minute before resuming... (From RateLimitError)")
                print(e)
                time.sleep(30)
                retry_count += 1

                if retry_count >= max_retries:
                    print("Exceeded maximum retries for parsing PDF.... (From RateLimitError)")
                    return response

        def extract_yoer_exact(data_dict):
            max_retries = 5
            retry_count = 0
            try:
                yoer_prompt_system = f"""[Instruction] 
                You will be provided with details such as the candidate's previous job roles. Please act as a hiring manager with 20 years experience to evaluate the candidate's previous job roles.
                1. Identify job roles that are specifically in {self.job_title}.
                2. Output only all of the duration of the specific job roles into a python list.
                3. The output format should strictly follow the format in the example provided.
                Example of the output: Total duration: [[2,3,4]]

                [Question]
                What are the job durations for the job roles that are specific to {self.job_title} in the candidate's previous job experience?
                """

                yoer_prompt_user = f"""
                Candidate's Previous Job Roles: {data_dict["Previous job roles"]}
                """
                client = AzureOpenAI()
                response = client.chat.completions.create(
                    model="gpt-35-turbo-16k", # 3.5 turbo 
                    messages=[
                        {"role": "system", "content": yoer_prompt_system},
                        {"role": "user", "content": yoer_prompt_user}
                    ],
                    temperature=0.3
                )
                print("Response from yoer",response)
                return response.choices[0].message.content
            except openai.RateLimitError as e:
                print(f"OpenAI rate limit exceeded. Pausing for one minute before resuming... (From RateLimitError)")
                print(e)
                time.sleep(30)
                retry_count += 1

                if retry_count >= max_retries:
                    print("Exceeded maximum retries for parsing PDF.... (From RateLimitError)")
                    return response

        def extract_duration(string):
            matches = re.findall(r'\[\[([0-9., ]+)\]\]', string)
            if matches:
                # Split by comma and directly convert each element to float
                list_of_floats = [float(x.strip()) for x in matches[0].split(",")]
                return list_of_floats
            else:
                print("No matches found for the pattern.")
                return []  # Fix to return a list directly

        def sum_floats_in_list(lst):
            if lst != 0:
                return math.fsum(lst)
            else:
                return 0

        def calculate_yoer(yoer_total, input_string, weightage):

            c_total_yr_exp = float(yoer_total)
            out_weighted_score = 0
            
            # Use parse_range to get the lower and upper limits and condition
            in_threshold_lower_limit, in_threshold_upper_limit, condition = self.parse_range(input_string)

            # Calculate the candidate's score based on their experience
            if c_total_yr_exp < in_threshold_lower_limit:
                out_weighted_score = 0  # does not meet requirement
            elif in_threshold_lower_limit <= c_total_yr_exp <= in_threshold_upper_limit:
                out_weighted_score = 1.0 * weightage  # within range ir equal 
            elif c_total_yr_exp > in_threshold_upper_limit:
                out_weighted_score = 0.5 * weightage  # overqualified
            else:
                out_weighted_score = 0


            return out_weighted_score

        if 'Years of experience in similar role' in index:
            response_yoer = extract_yoer_similar(data_dict)
            yoer_list = extract_duration(response_yoer)
            yoer_total = sum_floats_in_list(yoer_list)
            res = calculate_yoer(yoer_total, input, weightage)
            data_dict['Years of experience in similar role'] = yoer_total
            pos = list(data_dict.keys()).index('Years of experience in similar role')
            items = list(data_dict.items())
            items.insert(pos+1, ('Years of experience in similar role Score', res))
            data_dict = dict(items)
            print(f"Candidate: {data_dict['Name']}\t\t8. Yr of Exp in Role Score:{res}/{weightage}\t C: {yoer_total} E: {input}")
            return data_dict
        else:
            response_yoer = extract_yoer_exact(data_dict)
            yoer_list = extract_duration(response_yoer)
            yoer_total = sum_floats_in_list(yoer_list)
            res = calculate_yoer(yoer_total, input, weightage)
            data_dict['Years of experience in exact role'] = yoer_total
            pos = list(data_dict.keys()).index('Years of experience in exact role')
            items = list(data_dict.items())
            items.insert(pos+1, ('Years of experience in exact role Score', res))
            data_dict = dict(items)
            
            print(f"Candidate: {data_dict['Name']}\t\t8. Yr of Exp in Role Score:{res}/{weightage}\t C: {yoer_total} E: {input}")
            return data_dict

    # 11 
    def evaluate_current_location(self, data_dict, input, weightage):

        dataset_path = 'daerah-working-set.csv'
        city_data = pd.read_csv(dataset_path)

        def get_coordinates(city_name, country):
            # Try to get the coordinates from the dataset
            print("city name and country",city_name,country)
            try:
                city_info = city_data[city_data['Negeri'] == city_name]
                if country.lower() == "malaysia":
                    if city_info.empty==True:
                        city_info = city_data[city_data['Bandar'] == city_name]
                    latitude, longitude = city_info['Lat'].values[0], city_info['Lon'].values[0]
                    print("method1")
                    return latitude, longitude
            except IndexError:
                try:
                    http = urllib3.PoolManager(1, headers={'user-agent': 'cv_parser_geocoder'})
                    url = f'https://nominatim.openstreetmap.org/search?q={city_name}%2C+Malaysia&format=jsonv2&limit=1'
                    resp = http.request('GET', url)
                    loc = json.loads(resp.data.decode())
                    return loc[0]['lat'],loc[0]['lon']
                except:
                    return None,None
                

        def get_city_coast(latitude, longitude):
            east_coast_range =  (2.618, 6.2733, 101.3765, 103.6015)
            north_coast_range = (3.6857, 6.6999, 99.7166, 101.5265)
            middle_coast_range = (2.6884, 3.7801, 100.9878, 101.8911)
            south_coast_range =  (1.4645, 2.9702, 101.7863, 103.9107)
            east_malaysia_range = (1.0104, 6.9244, 109.7889, 119.0566)

            try:
                # Check which coast the city falls into
                if is_in_region(latitude, longitude, east_malaysia_range):
                    return "East Malaysia"
                elif is_in_region(latitude, longitude, middle_coast_range):
                    return "Middle Coast"
                elif is_in_region(latitude, longitude, east_coast_range):
                    return "East Coast"
                elif is_in_region(latitude, longitude, north_coast_range):
                    return "North Coast"
                elif is_in_region(latitude, longitude, south_coast_range):
                    return "South Coast"
                else:
                    return "Out of Malaysia"
            except TypeError:
                return "Location Not Detected"

        def is_in_region(latitude, longitude, region_range):
            min_lat, max_lat, min_lon, max_lon = region_range
            return min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon
        
        state_mapping = {'wilayah persekutuan': 'WP', 'selangor': 'Selangor', 'johor': 'Johor', 'penang': 'Penang', 'pulau pinang': 'Penang', 'sabah': 'Sabah', 'sarawak': 'Sarawak', 'perak': 'Perak', 'kedah': 'Kedah', 'pahang': 'Pahang', 'terengganu': 'Terengganu', 'kelantan': 'Kelantan', 'negeri sembilan': 'N.Sembilan', 'melaka': 'Melaka','melacca': 'Melaka','perlis': 'Perlis'}
        
        def clean_state(data_dict):
            try:
                for key, value in state_mapping.items():
                    if key.lower() in data_dict['Candidate Current Location'][0]['State'].lower():
                        data_dict['Candidate Current Location'][0]['State'] = value
                        break
                return data_dict
            except:
                return data_dict

        def clean_location_string(location_str):
            try:
                # Split the string into city and country
                location_parts = list(map(str.strip, location_str.split(',')))

                # Handle the case when location_str only has city and country
                if len(location_parts) == 2:
                    state, country = location_parts

                    for key, value in state_mapping.items():
                        if key.lower() in state.lower():
                            state = value
                            break

                    city = 'N/A'
                elif len(location_parts) == 3:
                    city, state, country = location_parts

                    for key, value in state_mapping.items():
                        if key.lower() in state.lower():
                            state = value
                            break
                else:
                    country = location_parts[0]
                    state = 'N/A'
                    city = 'N/A'

                # Create the result dictionary
                result = {'Country': country, 'State': state, 'City': city}

                return result
            except ValueError:
                return location_str
        
        def evaluate_coordinate(cleaned_location,data_dict):
            #Get coordinates for required location and candidate location
            latitude1, longitude1 = get_coordinates(cleaned_location['State'],cleaned_location['Country'])
            print(latitude1, longitude1)
            latitude2, longitude2 = get_coordinates(data_dict['Candidate Current Location'][0]['State'], data_dict['Candidate Current Location'][0]['Country'])
            print(latitude2, longitude2)
            #Define the coast of required location and candidate location
            coast1 = get_city_coast(latitude1, longitude1)
            coast2 = get_city_coast(latitude2, longitude2)
            #Located at the same region(coast)
            if coast1 == coast2:
                return weightage*0.5
            #Located at different region
            else:
                return 0


        def evaluate_location(cleaned_location,data_dict,weightage):
            try:
                print(cleaned_location)
                print(data_dict['Candidate Current Location'])
                # If candidate is in Malaysia
                if cleaned_location['Country'].lower() == "malaysia" and data_dict['Candidate Current Location'][0]['Country'].lower() == "malaysia":
                    # If Option 1 in excel
                    if cleaned_location['State'].lower() == 'n/a' and cleaned_location['City'].lower() == 'n/a':
                        return weightage
                    
                    # If same state
                    elif (data_dict['Candidate Current Location'][0]['State'].lower() == cleaned_location['State'].lower()):
                        # State = N/A
                        if cleaned_location['State'].lower() == 'n/a':
                            if cleaned_location['City'].lower() == 'n/a':
                                return 0
                            else:
                                print("weightage here")
                                return weightage
                        # State != N/A
                        else:
                            return weightage
                        
                    # if not same state
                    elif (data_dict['Candidate Current Location'][0]['State'].lower() != cleaned_location['State'].lower()):
                        # same city
                        if (data_dict['Candidate Current Location'][0]['City'].lower() == cleaned_location['City'].lower() == "N/A"):
                            return 0
                        else:
                            return evaluate_coordinate(cleaned_location,data_dict)
                        
                    # if same city
                    elif (data_dict['Candidate Current Location'][0]['City'].lower() == cleaned_location['City'].lower()):
                        # City = N/A
                        if cleaned_location['City'].lower() == 'n/a':
                            return 0
                        else:
                            print("weightage here")
                            return weightage
                    else:
                        return 0
                        
                # If candidate is overseas
                else:
                    if data_dict['Candidate Current Location'][0]['Country'] == cleaned_location['Country']:
                        print(cleaned_location['Country'],data_dict['Candidate Current Location'][0]['Country'])
                        return weightage
                    else:
                        return 0
            except TypeError as e:
                print("Different Country detected")
                print(e)
                return 0

        # Example usage:
        cleaned_location = clean_location_string(input)
        cleaned_dict = clean_state(data_dict)
        out_location_score =  evaluate_location(cleaned_location,cleaned_dict,weightage)
        pos = list(data_dict.keys()).index('Candidate Current Location')
        items = list(data_dict.items())
        items.insert(pos+1, ('Candidate Current Location Score', out_location_score))
        data_dict = dict(items)
        print("data dict at location",data_dict)
        print (f"Candidate: {data_dict['Name']}\t\t 11. Location Score: {out_location_score}/{weightage}\t  E:{cleaned_location} C: {data_dict['Candidate Current Location']}\n")
        return data_dict


    # 12 
    def evaluate_targetted_employer (self, data_dict, in_target_employer, in_weightage_employer): 
        criterion_key, out_targetted_employer_score = "Targeted Employer", 0 # for this criterion, add "Targeted Employer" and "Targeted Employer Score" since not in data_dict

        # initialising key at position 
        pos = list(data_dict.keys()).index("Previous job roles")
        items = list(data_dict.items())
        items.insert(pos + 1, (f"{criterion_key}", ""))
        items.insert(pos + 2, (f"{criterion_key} Score", ""))
        data_dict = dict(items)
        print (f"after assigning in eval targ emp: {data_dict}")

        # parse into include and excluded target comapanies 
        included_input = []
        excluded_input = []
        exclusion_match = ""

        def validate_input_format(input_string): 
            """
            Check if CVMATCHING template format correct for 
            Example: 
                True - "include(Shell, BP) ,  exclude( KLCC, Novella Clinical, Fidelity Investments)    
                True - "include(), exclude()"   
                True - "include(Shell, BP) , exclude()"    
                True - "include() , exclude(Shell, BP)" 
                False -  "include() , exclude(Shell, " 

            """
            # Regular expression pattern to match the valid format
            # pattern = r'^(include\([\w\s,]*\)\s*,\s*exclude\([\w\s,]*\)\s*)+$'
            pattern = r'^(include\((.*?)\)\s*,\s*exclude\((.*?)\)\s*)+$' # include special characters in company names eg &.
            
            # Check if the input string matches the pattern
            if re.match(pattern, input_string):
                return True
            else:
                return False
            
        def parse_targemp_input (correct_format_inputstring):    
            '''
                include_input updates to space-removed values 
                excluded_input updates to space-removed values 

                Example: 
                included_input: ['PetronasDigital']
                excluded_input: ['KLCC', 'NovellaClinical', 'FidelityInvestments']

                Reasoning: 
                More robust matching when spaces are removed. ExxonMobil matches Exxon Mobil inputted by User 
            ''' 
            # Regular expression pattern to match the include and exclude sections
            pattern = r'(include|exclude)\((.*?)\)'
            matches = re.findall(pattern, correct_format_inputstring)

            for match in matches:
                action, values = match
                values_list = [value.strip().replace(" ", "") for value in values.split(',')]
                
                if action == 'include':
                    included_input.extend(values_list)
                elif action == 'exclude':
                    excluded_input.extend(values_list)
            return True 

        # Preprocessing input & resume employers 
        def clean_employer_lst(input_str):
            """
            removes common words for better string matching 
            """
            # List of common words to remove
            common_words_to_remove = ["sdn", "bhd", "berhad", "ptd", "ltd", "inc", "co", "llc", "or", "and", "&"]
            pattern = r'\b(?:' + '|'.join(re.escape(word) for word in common_words_to_remove) + r')\b|-|\s+'
            cleaned_str = re.sub(pattern, '', input_str, flags=re.IGNORECASE)
            cleaned_list = [word.strip() for word in cleaned_str.split(',')]
            return cleaned_list
        
        def extract_indsutries (gpt_response): 
            """
            Extract industries from customised gpt output response. Example: 
                gpt_response = "[[Marketing, Food & Beverage, Shipping, Fashion, Cosmetics]]"
                output = ["Marketing", "Food & Beverage", "Shipping", "Fashion", "Cosmetics"]
            """
            # Ensure input is a string and follows the expected format
            if not isinstance(gpt_response, str) or not gpt_response.startswith("[[") or not gpt_response.endswith("]]"):
                return ["Unknown"]

            # Extract the content inside the outer brackets and split by comma
            # The slice [2:-2] removes the outermost brackets "[[" and "]]"
            industries = [industry.strip() for industry in gpt_response[2:-2].split(',')]

            return industries
        def get_employer_industries_gpt4 (company_name, company_location = ""): 
            max_retries = 5
            retry_count = 0
            # Classify employer industry by gpt
            system_p = f"""You are a helpful assistant. Given a company name and details, your task is to classify the given company's industry it is involve in as per The International Labour Organization.
            1. Classify the industries the company falls into according to The International Labour Organization, based on the company. 
            2. Output only all of industries in python list.
            3. The output format should strictly follow the format in the example provided below - enclosed with double brackets, comma-seperated
            4. A company can be classified in more than 1 industries. 
            Example of the output:
                example 1:  [[Marketing, Food & Beverage, Shipping, Fashion, Cosmetics]]
                example 2: [[Finance]]
                example 3: [[Unknown]] if the company is unfamiliar or you are unsure, output this. 

            """
            in_target_employer_petronas_description = "Petronas is a Malaysian oil and gas company that is involved in upstream and downstream activities. It is the largest oil and gas company in Malaysia, with operations in more than 30 countries around the world. Petronas is involved in exploration, production, refining, marketing, trading, and distribution of petroleum products. It also has interests in petrochemicals, shipping, engineering services, power generation, and other related businesses."
            p_example = f'[The Start of Company description] {in_target_employer_petronas_description}[The End of Company description] '
            p_example_response_format = "[[Oil and Gas, Petrochemicals, Refining, Retail, Shipping, Exploration and Production, Engineering and Construction]]"
            

            p = f'Classify the industries according to The International Labour Organization of the given company. Return results in the aforementioned output format. Given Company: {company_name}, located at {company_location}'
            try:
                client = AzureOpenAI()
                response = client.chat.completions.create(
                    model="gpt-35-turbo-16k", 
                    messages=[
                        {"role": "system", "content": system_p},
                        {"role": "user", "content": p_example},
                        {"role": "assistant", "content": p_example_response_format},
                        {"role": "user", "content": p}
                    ]
                )
                print("Response from edu",response)
                try:
                    result = response.choices[0].message.content
                    industries_lst = extract_indsutries (result) if result else None 
                    print (f'GPT response on industry: {result}\tEXTRACTED INDUSTRIES: {industries_lst}')
                    return industries_lst
                except KeyError:
                    return "undectected"
            except openai.RateLimitError as e:
                print(f"OpenAI rate limit exceeded. Pausing for one minute before resuming... (From RateLimitError)")
                print(e)
                time.sleep(30)
                retry_count += 1

                if retry_count >= max_retries:
                    print("Exceeded maximum retries for parsing PDF.... (From RateLimitError)")
                    return response
            except Exception as ire:
                print("InvalidReqError",ire)
                return "undetected"
            
        def get_employer_industries(company_name):
            '''
                Returns the classified list of industries the target employer is possibly in 
                according to The International Labour Organization 

                Usage: 
                - more precise and broad classification of industries given a company name 
                - assigning more accurate industries to data_dict employers for future use 
                
            '''

            # USER_AGENTS = [
            # "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
            # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
            # # Add more user agents as needed
            # ]
            # proxy_pool = cycle(PROXIES)
            # user_agent_pool = cycle(USER_AGENTS)

            # def get_random_user_agent():
            #     return random.choice(USER_AGENTS)

            def google_search(company_name):
                retry_count = 0
                max_retries = 5
                google_search_employer_desc_result = ""
                while retry_count < max_retries:
                    try:
                        # Get company company description
                        query = company_name + " company About Us"
                        print(f"Googling company description: {query}")
                        google_results = search(query, num_results=2, advanced=True, lang='en')

                        for result in google_results:
                            google_search_employer_desc_result += result.description
                            # print(f"\n\tGoogle: {google_search_employer_desc_result}")                    
                            print(f"\tGoogling: {query}")
                        return google_search_employer_desc_result 
                    except Exception as http_err:
                            print(f"HTTP error 429 (Too Many Requests) occurred. Retrying...")
                            retry_count += 1
                            time.sleep(10)  # Wait for 60 seconds before retrying
                return google_search_employer_desc_result+company_name
                            
                            
            google_search_employer_desc_result = google_search(company_name)

            # Classify employer industry by gpt
            system_p = f"""you are a helpful assistant that accurately classifies company to their industries.You will be provided with details such as the candidate's previous job roles.
            1. Cassify the industries the company falls into according to The International Labour Organization, based on the following search results of their company profile. Return in comma-separated values in a string.
            2. Output only all of industries in python list.
            3. The output format should strictly follow the format in the example provided - enclosed with double brackets, comma-seperated
            Example of the output:
                example 1:  [[Marketing, Food & Beverage, Shipping, Fashion, Cosmetics]]
                example 2: 
            """
            in_target_employer_petronas_description = "Petronas is a Malaysian oil and gas company that is involved in upstream and downstream activities. It is the largest oil and gas company in Malaysia, with operations in more than 30 countries around the world. Petronas is involved in exploration, production, refining, marketing, trading, and distribution of petroleum products. It also has interests in petrochemicals, shipping, engineering services, power generation, and other related businesses."
            p_example = f'[The Start of Company description] {in_target_employer_petronas_description}[The End of Company description] '
            p_example_response_format = "[[Oil and Gas, Petrochemicals, Refining, Retail, Shipping, Exploration and Production, Engineering and Construction]]"
            

            p = f'Classify the industries the following company falls into according to The International Labour Organization, based on the following search results of their company profile. Return in comma-separated values in a string. Company: {google_search_employer_desc_result}'
            try:
                client = AzureOpenAI(
                    azure_deployment = "gpt-35-turbo-16k"
                )
                response = client.chat.completions.create(
                    model="gpt-35-turbo-16k", 
                    messages=[
                        {"role": "system", "content": system_p},
                        {"role": "user", "content": p_example},
                        {"role": "assistant", "content": p_example_response_format},
                        {"role": "user", "content": p}
                    ]
                )
                print("Response from edu",response)
                try:
                    result = response.choices[0].message.content
                    industries_lst = extract_indsutries (result) if result else None 
                    print (f'GPT response on industry: {result}\tEXTRACTED INDUSTRIES: {industries_lst}')
                    return industries_lst
                except KeyError:
                    return "undetected"
            except Exception as ire:
                print("InvalidReqError",ire)
                return "undetected"

        
        def check_if_matching_employer_industry():
            '''
                Input: 
                    user_input_bool: True if input is a list (ie from CVMatching xlsx since can be >1 company)
                    in_target_employer: Company Name
                Used when candidate has not work in target employer specified, check for matching industries: 
                    1. Use googlesearch to search for company background/description of target employer
                    2. Ask GPT to classify the industries based on this description 
                    3. Check against candidate's previous job company industries
                    4. if candidate worked in similar industries: 50%, else 0%
            '''
            # variables
            global gth_criteria_total_score  # Declare as global
            out_targetted_employer_score =  0


            if (self.targEmp_industries_included == []): 
                init_input_employer_industry()

            candidate_industries = data_dict["Industries"]
            
            # find matches between overall industries and included()
            list1 = [x.lower().replace(" ", "") for x in candidate_industries if x.lower().replace(" ", "") != "unknown"]   
            list2 = [x.lower().replace(" ", "") for x in self.targEmp_industries_included if x.lower().replace(" ", "") != "unknown"]
            matches = [x for x in list1 if x in list2]

            print(f"GPT-ed Classified Industries.\t Included:{included_input} \tExcluded {excluded_input}. Included Industries{self.targEmp_industries_included}\t Candidate's data_dict['Industries']: {candidate_industries}\t Matched industries are: {matches}")
            if matches:
                print (f"Candidate: {data_dict['Name']}\t\t 12. Targeted Employer Score: {out_targetted_employer_score}/{in_weightage_employer}\t Result: Case 2: Matching Industries are {matches}\n")
                data_dict[f'{criterion_key}'] =  f"Matching industries detected: {matches}"
                data_dict[f'{criterion_key} Score'] =  0.5*float(in_weightage_employer)
                return data_dict
            else:
                print (f"Candidate: {data_dict['Name']}\t\t 12. Targeted Employer Score: {out_targetted_employer_score}/{in_weightage_employer}\t Result: Case 3: NO MATCHING INDUSTRY \n ")
                data_dict[f'{criterion_key}']  =f"No exact match and no matching industry from past employers detected" 
                data_dict[f'{criterion_key} Score'] =  0 
                return data_dict

        def worked_in_excluded(candidate_employers, excluded): 
            excluded_matches = []
            for x in candidate_employers:
                    if x in excluded: 
                    #  excluded_matches = f"WARNING: Candidate: {data_dict['Name']} 12. Targeted Employer: Candidate works in a company in 'excluded' company: {x} "
                        excluded_matches = f"Exclusion detected[{x}]"
                        return excluded_matches, True 
            return excluded_matches, False      
        
        def init_input_employer_industry ():
            """
            Initialises list for related-industries in criteria file by user
            
            """ 
            target_employer_industries = set()
            for employer in included_input:
                print(f"xlsx included employer {employer}")
                if employer: 
                    a = get_employer_industries_gpt4(employer)
                    target_employer_industries.update(a)
            # Assuming 'target_employer_industries_lst' is a list of lists (each inner list contains industries for an employer)
            self.targEmp_industries_included = list (target_employer_industries)
            print (f"RESUME PARSER CLASS INTIALISED: XLSX Target Employer related googlesearch industries.\tIncluded {included_input}\t self.targEmp_industries: {self.targEmp_industries_included}\n")
            return True 
                    
        # User/Employer Template input validation
        try:
            # Assuming validate_input_format raises an exception if validation fails
            if not validate_input_format(in_target_employer):
                raise ValueError("CVMatching Template.xlsx input string is invalid at 12.Target Employer")

            # If validation passes, proceed with parsing
            parse_targemp_input(in_target_employer)  # included, excluded is updated
            print(f"included: {included_input}, \t excluded: {excluded_input}")
        except ValueError as e:
            # Handle the validation error
            error_message = f"Warning, 12. Target Employer in file CVMatching Template.xlsx {e}"
            print(error_message)
            # self.targEmp_exclusion_matched = "CVMatching Template.xlsx input string is invalid at 12.Target Employer"
            return -1
            
        # Preprocessing inputs 
        req_employers = clean_employer_lst("".join(included_input))  # clean input from excel from common words 
        candidate_employers = clean_employer_lst(",".join([role["Job Company"] for role in data_dict["Previous job roles"] if isinstance(role, dict)]))
        print(f"12. Evaluating Target Employer\tIncluded: {included_input} \t excluded: {excluded_input}\tCandidate's previous employers: {candidate_employers}")

        # Preprocessing Data_dict of candidate: Reassign GPT classified industries for candidate's each previous employer 
        overall_industries = set()
        for x in data_dict["Previous job roles"]:
            if isinstance(x, dict):
                q = x['Job Company'] 
                l = x["Job Location"] 
                industries_list = get_employer_industries_gpt4(q, l)  # This now returns a list directly
                
                # Directly assign the list without splitting
                x["Industries"] = industries_list
                
                # Update overall industries without needing to split; handle single-value lists correctly
                overall_industries.update([j.strip().strip('.') for j in industries_list])
                
                # Adjust the print statement to directly use industries_list
                print(f"{q} located at {l} is gpt-classified as a company in industries: {industries_list}")
                
        data_dict["Industries"] = list(overall_industries)
        # Scoring Method
        # 1. Check if candidate work in excluded companies 
        exclusion_match, excluded_flag = worked_in_excluded(candidate_employers, excluded_input)
        if excluded_flag: 
            print("it did print!!!",exclusion_match)
            data_dict[f'{criterion_key}'] =  str(exclusion_match)
            data_dict[f'{criterion_key} Score'] =  0
                
            return data_dict
        else:
            # 2. Check for exact match with cleaned lists (employer and user)
            matched_employer = []
            for candidate in candidate_employers:
                # Skip if candidate is empty or whitespace
                if not candidate.strip():
                    continue

                for required in req_employers:
                    if re.search(fr'\b{re.escape(candidate)}\b', required, re.IGNORECASE):
                        matched_employer.append(candidate)
                        break # breaks right after matching 1 employer

            # if matched_employer:
            #     print (f"Candidate: {data_dict['Name']}\t\t 12. Targeted Employer Score: {out_targetted_employer_score}/{in_weightage_employer}\t  Result: Case 1: MATCHING EMPLOYER \t Matches = {matched_employer}\n)")
            #     data_dict['Targeted Employer'] =  f"Inclusion detected: {matched_employer}"
            #     data_dict["Targeted Employer Score"] = in_weightage_employer

            # else: # 3: Check for related industry in candidate's past employers 
            #     print ('\t...12. Target Employer: Checking for any past employers matching to industry of target employer')
            #     data_dict = check_if_matching_employer_industry()
                    
            if not matched_employer:# 3: Check for related industry in candidate's past employers 
                print ('\t...12. Target Employer: Checking for any past employers matching to industry of target employer')
                data_dict = check_if_matching_employer_industry()
                
            else: # exact match employer 
                print (f"Candidate: {data_dict['Name']}\t\t 12. Targeted Employer Score: {out_targetted_employer_score}/{in_weightage_employer}\t  Result: Case 1: MATCHING EMPLOYER \t Matches = {matched_employer}\n)")
                data_dict[f'{criterion_key}'] =  f"Inclusion detected: {matched_employer}"
                data_dict[f'{criterion_key} Score'] = float(in_weightage_employer)
        return data_dict

    # 14 
    def evaluate_language_score(self, data_dict, input, weightage):
        print("language now")
        criterion_key = "Language"
        match_percentage, rounded_score = 0,0
        try:
            custom_languages = ["Bahasa Melayu", "Bahasa Malaysia", "Malay"]
            def check_custom_languages(input_list):
                return set(lang.lower()for lang in custom_languages if lang.lower() in input_list)
            
            languages_str = ', '.join(data_dict['Language'])

            nlp = spacy.load('en_core_web_md')
            
            def get_lang_detector(nlp, name):
                return LanguageDetector()

            if languages_str == "N, /, A" or languages_str == "N/A":
                Language.factory("language_detector", func=get_lang_detector)
                nlp.add_pipe('language_detector', last=True)
                doc1 = nlp(str(data_dict))
                if doc1._.language['language']=='en':
                    languages_str='English'
                    data_dict['Language'] = ['English']
            doc1 = nlp(languages_str)
            doc2 = nlp(input)
            
            languages1 = set(ent.text.strip() for ent in doc1.ents if ent.label_ == "LANGUAGE")
            print(languages1)
            languages2 = set(ent.text.strip() for ent in doc2.ents if ent.label_ == "LANGUAGE")
            print(languages2)

            languages1.update(check_custom_languages(languages_str))
            languages2.update(check_custom_languages(input))

            matched_languages = set(l.lower() for l in languages1).intersection(set(l.lower() for l in languages2))

            # Calculate the percentage of matches
            if languages1:
                match_percentage = len(matched_languages) / len(languages2) * 100
            else:
                match_percentage = 0
            language_score = match_percentage/100*weightage
            rounded_score = round(language_score)
            #print (f"Candidate: {data_dict['Name']}\t\t 14. Language Score: {rounded_score}/{weightage}\t C:{languages_str}, E: {input} \n")
            
            # update data_dict
            pos = list(data_dict.keys()).index(criterion_key)
            items = list(data_dict.items())
            items.insert(pos+1, (f'{criterion_key} Score', rounded_score))
            data_dict = dict(items)
            return data_dict
            
        except Exception as e:
            print("Error on language",e)
            traceback.print_exc()  # This will print the traceback information
            # update data_dict
            pos = list(data_dict.keys()).index(criterion_key)
            items = list(data_dict.items())
            items.insert(pos+1, (f'{criterion_key} Score', 0))
            data_dict = dict(items)
            return data_dict
            

    # 15 
    def evaluate_salary_score(self, data_dict, input, weightage):
        print("salary now")
        criterion_key = "Expected Salary in RM"
        """
        Checks if the candidate's expected salary matches the employer's range.

        Args:
        in_salary (str): Employer's expected salary range.
        c_exp_salary (str): Candidate's expected salary.

        Returns:
        int: Score indicating the match percentage.
        """
        # Assign 0 score for N/A or empty values
        if  input in ("N/A", "", "-") or data_dict['Expected Salary in RM'] in ("N/A", "") :
            out_salary_score = 0
        else: 
            # Parse employer's expected salary range
            # print ("Employer")
            in_exp_sal_llimit, in_exp_sal_ulimit, in_exp_sal_condition = self.parse_range(input)

            # Parse candidate's expected salary, calculate average if it's a range
            c_exp_sal = 0 # default is 0 
            # print ("Candidate")
            c_exp_sal_llimit, c_exp_sal_ulimit, c_exp_sal_condition = self.parse_range(data_dict['Expected Salary in RM'])
            if c_exp_sal_llimit != c_exp_sal_ulimit:
                # Alternative: Calculate average for a range
                    # c_exp_sal = (c_exp_sal_llimit + c_exp_sal_ulimit) / 2  
                c_exp_sal = c_exp_sal_llimit # assume lower limit when cv states sal range for now 
            else:
                c_exp_sal = c_exp_sal_llimit  # Use lower limit as single value if cv input not a range
            # print (f"\tCandidate expected salary: {c_exp_sal}")

            # Check if the candidate's expected salary falls within the employer's range
            if in_exp_sal_llimit <= c_exp_sal <= in_exp_sal_ulimit:
                res = 1  # 100% 
            else:
                # Extra:
                # # Calculate a score based on how close the candidate's salary is to the employer's range
                # range_width = in_exp_sal_ulimit - in_exp_sal_llimit
                # if out_exp_sal_score < in_exp_sal_llimit:
                #     res = max(0, 50 - ((in_exp_sal_llimit - out_exp_sal_score) / range_width) * 50)  # Decrease score as it's below range
                # else:
                #     res = max(0, 50 - ((out_exp_sal_score - in_exp_sal_ulimit) / range_width) * 50)  # Decrease score as it's above range
                res = 0
            
            out_salary_score = res * weightage

        # update data_dict
        pos = list(data_dict.keys()).index(criterion_key)
        items = list(data_dict.items())
        items.insert(pos+1, (f'{criterion_key} Score', out_salary_score))
        data_dict = dict(items)
        print (f"Candidate: {data_dict['Name']}\t\t 15. Exp Salary in RM Score: {out_salary_score}\t Employer: {input}, Candidate: {data_dict['Expected Salary in RM']}\n ")

        return data_dict
    #10 
    def evaluate_prof_cert_phrase(self, data_dict, input, weightage):
        print("pro cert now")
        # Read the abbreviation CSV file
        file_path = os.path.join(os.path.dirname(__file__), 'CVMatching_Prof_Cert_Wikipedia.xlsx')
        abb_csv = pd.read_excel(file_path)
        abb_csv = abb_csv[['Name', 'Abbreviation']]
        abb_csv = abb_csv.dropna(subset=['Abbreviation']).reset_index(drop=True)

        abb_csv['Name_lower'] = abb_csv['Name'].str.lower()
        unique_elements = [ue.strip() for ue in input.split(",")]

        # Retrieve 'Professional Certificate' field from data_dict
        professional_certificates = set(data_dict.get('Professional Certificate', []))
        print("printing resume profc:",professional_certificates)
        for phrase in unique_elements.copy():
            # Convert the current phrase to lowercase for case-insensitive comparison
            phrase_lower = phrase.lower()
            
            # Check if the lowercase phrase is an exact match in any lowercase entry in the 'Name' or 'Abbreviation' columns
            match = abb_csv[(abb_csv['Name_lower'] == phrase_lower) | (abb_csv['Abbreviation'].str.lower() == phrase_lower)]
            
            # If there is a match, remove both the abbreviation and the full name from the unique elements
            if not match.empty:
                # Update with matched abbreviations and names
                unique_elements.append([match['Name'].values[0],match['Abbreviation'].values[0]])
                unique_elements.remove(phrase)

        def detect_match_phrases(resume, match_phrases):
            matches = []
            for phrase in match_phrases:
                print(phrase)
                if type(phrase) == list:
                    for x in phrase:
                        print("printing x",x)
                        pattern = re.compile(fr'\b{re.escape(x)}\b', re.IGNORECASE)
                        matches.extend(pattern.findall(resume.lower()))
                        print("printing matches",matches)
                else:
                    # Use case-insensitive matching and convert to lowercase
                    pattern = re.compile(fr'\b{re.escape(phrase)}\b', re.IGNORECASE)
                    matches.extend(pattern.findall(resume.lower()))

            # Remove duplicates by converting the list to a set and back to a list
            unique_matches = list(set(matches))

            return unique_matches
        
        def evaluate_candidate_score(matched_phrases, match_phrase_input, weightage):
            # Calculate the score based on the weightage
            score = round(len(matched_phrases) / len(match_phrase_input) * weightage,2)

            return score

        
        # Retrieve 'Professional Certificate' field from data_dict
        professional_certificates = data_dict.get('Professional Certificate', [])

        # Convert data_dict to a string
        data_dict_str = ' '.join(professional_certificates)

        # Detect matched phrases
        matched_phrases = detect_match_phrases(data_dict_str, unique_elements)
        print('matched_phrases', matched_phrases,len(matched_phrases))
        print('unique_elements', unique_elements,len(unique_elements))
        # Evaluate candidate score
        score = evaluate_candidate_score(matched_phrases, unique_elements, weightage)
        
        pos = list(data_dict.keys()).index('Professional Certificate')
        items = list(data_dict.items())
        items.insert(pos+1, ('Professional Certificate Score', score))
        data_dict = dict(items)
        
        print (f"Candidate: {data_dict['Name']}\t\t 10. Prof Cert Score: {score}/{weightage}\t Employer's Certs: {input},  Candidate's Certs: {professional_certificates}\n ")
        
        return data_dict

    def evaluate_year_grad_score(self, data_dict, input_year, weightage): 
        print('Evaluating year of graduation now')
        criterion_key, out_yr_grad  = "Years of Graduation", 0 

        # update data_dict
        pos = list(data_dict.keys()).index("Education Background Score")
        items = list(data_dict.items())
        items.insert(pos+1, (f'{criterion_key}', ""))
        items.insert(pos+2, (f'{criterion_key} Score', 0))
        data_dict = dict(items)

        if 'Education Background' not in data_dict or not isinstance(data_dict['Education Background'], list):
            print("No educational background provided.")
            out_yr_grad = 0 
            data_dict[f'{criterion_key}'] = "No educational background provided."
            data_dict[f'{criterion_key} Score' ] = out_yr_grad
        else: 
            # Sort education background by year of graduation once, after preprocessing
            data_dict['Education Background'].sort(key=lambda x: x.get('Year of Graduation', ''), reverse=True)
            # Preprocess and validate year of graduation entries
            res = ""
            for edu in data_dict['Education Background']:
                if isinstance(edu, dict) and 'Year of Graduation' in edu:
                    year_of_graduation = str(edu['Year of Graduation'])  # Ensure it's a string for comparison
                    if not year_of_graduation.isdigit() and year_of_graduation.lower() not in ['present', 'current']:
                        edu['Year of Graduation'] = 'N/A'
                    elif year_of_graduation.lower() in ['present', 'current']:
                        edu['Year of Graduation'] = 'Still Studying'

                    year_of_graduation = str(edu.get('Year of Graduation')) 
                    res += year_of_graduation + ", "
                    if year_of_graduation == input_year:
                        out_yr_grad = weightage


            # # Evaluate year of graduation score
            # for edu in data_dict['Education Background']:
            #     if isinstance(edu, dict):
            #         year_of_graduation = str(edu.get('Year of Graduation')) 
            #         res += year_of_graduation + ", "
            #         if year_of_graduation == input_year:
            #             out_yr_grad = weightage

            # Print the result
            res = res if res else "Not Specified" 
            data_dict[f'{criterion_key}'] = res
            data_dict[f'{criterion_key} Score'] = out_yr_grad
            candidate_name = data_dict.get('Name', 'Unknown')
            print(f"Candidate: {candidate_name}\t\t 16. Year of Grad: {out_yr_grad}\t Employer: {input_year},  Candidate: {res}")

        return data_dict



        