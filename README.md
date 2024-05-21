# Generative AI Resume Parser

## 1. Who Am I?

**Question:** What is the purpose of the Resume Parser tool?

**Answer:** The Resume Parser tool is designed to efficiently extract and evaluate key information from resumes. It helps streamline the recruitment process by automatically analyzing resumes against specified criteria, such as education background, work experience, skills, and more. The tool also provides a GPT-generated recommendation summary for each candidate, enhancing the evaluation process by offering insights based on the extracted data.

## 2. What are the expected outputs for the extracted results?

**Question:** What information does the Resume Parser tool extract from a candidate's resume?

**Answer:** The Resume Parser tool extracts the following information from a candidate's resume:

- **Name:** The candidate's full name.
- **Phone Number:** The candidate's contact phone number.
- **Email:** The candidate's email address.
- **Local:** Whether the candidate is Malaysian or not (Yes or No).
- **Expected Salary:** The salary the candidate expects, given in Malaysian Ringgit (RM).
- **Current Location:** The candidate's current location, including country, state, and city.
- **Education Background:** Details about the candidate's education, including field of study, degree level, CGPA, university name, start date, and graduation year.
- **Professional Certificates:** Any professional certificates the candidate holds.
- **Skill Groups:** The different skill sets the candidate possesses, or any technology tools, programs, or systems the candidate has experience with..
- **Languages:** Languages the candidate can speak or write.
- **Previous Job Roles:** Information about the candidate's past job roles, including job titles, companies, industries, start and end dates, job locations, and the duration of each job in years.

## 3. What is the format for the evaluating criteria details that users should follow?

**Question:** How should users format the details for evaluating criteria?

**Answer:** Users should follow the below formats for evaluating criteria details:

| Criteria                         | Format                                               | Example Input                                              |
|----------------------------------|------------------------------------------------------|------------------------------------------------------------|
| Education Background             | [Enter preferred education background]               | bachelor's or master's degree in data science              |
| Academic Result (CGPA)           | [Enter minimum CGPA]                                 | 3.85                                                       |
| Skill Groups                     | [Value], [Value], [Value]                            | Project Management, Coding                                 |
| Years of Total Work Experience   | "Option 1: Number<br>Option 2: Number-Number<br>Option 3: >Number<br>Option 4: <Number" | 20                                                         |
| Match Phrase                     | [Value], [Value], [Value]                            | petronas digital, spearhead                                |
| Technology (Tools, Program, System) | [Value], [Value], [Value]                            | Word, Excel, PowerPoint                                    |
| Years of Experience in Exact Role| "Option 1: Number<br>Option 2: Number-Number<br>Option 3: >Number<br>Option 4: <Number" | 5-10                                                       |
| Professional Certificate         | [Value], [Value], [Value]                            | ACCA, CFA I, CFA II                                        |
| Candidate Current Location       | "Option 1: Country<br>Option 2: State, Country<br>Option 3: City, State, Country" | "Option 1: Canada<br>Option 2: Sarawak, Malaysia<br>Option 3: Kuala Lumpur, Wilayah Persekutuan, Malaysia" |
| Targeted Employer                | include(Value, Value), exclude(Value, Value)         | include(Shell, BP), exclude(KLCC, ABC)                     |
| Age                              | "Option 1: Number<br>Option 2: Number-Number<br>Option 3: >Number<br>Option 4: <Number" | >50                                                        |
| Language                         | [Value], [Value], [Value]                            | English, Mandarin, Arabic                                  |
| Expected Salary in RM            | "Option 1: Number<br>Option 2: Number-Number<br>Option 3: >Number<br>Option 4: <Number" | <500000                                                    |
| Years of Graduation              | [Enter year of graduation]                           | 2024                                                       |

---

