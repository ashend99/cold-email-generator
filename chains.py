import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm =  ChatGroq(
                        groq_api_key = os.getenv("GROQ_API_KEY"),
                        model = 'llama-3.3-70b-versatile',
                        temperature = 0
                    )
    
    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
                        """
                        ### SCRAPPED TEXT FROM WEBSITE:
                        {page_data}
                        ### INSTRUCTION
                        The scrapped text is from teh career's page of a website.
                        Your job is to extract the job postings and retunr them in JSON format containing
                        following keys: 'role', 'experience', 'skills' and 'description'
                        Only retunr the valid JSON
                        ### VALID JSON (NO PREAMBLE):
                        """
                        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={'page_data':cleaned_text})

        try:
            json_parser = JsonOutputParser()
            res = json_parser.invoke(res.content)
        except OutputParserException:
            raise OutputParserException("Content too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]
    
    def write_mail(self, job, links):
        prompt_email =  PromptTemplate.from_template(
                        """
                        ### JOB DESCRIPTION:
                            {job_description}
                            
                            ### INSTRUCTION:
                            You are Ashen, a BSc graduate with a major in Computer Science. You are currently working as a AI Engineer with 3 years of experience. You are currently looking for a job change.  
                            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of yourself
                            in fulfilling their needs.
                            Also add the most relevant ones from the following links to showcase your portfolio: {link_list}
                            Remember you are Ashen, a BSc graduate with 3 years of experience in AI Engineering. 
                            Do not provide a preamble.
                            ### EMAIL (NO PREAMBLE):
                        """
                        )
                            #         You are Mohan, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
                            # the seamless integration of business processes through automated tools. 
                            # Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
                            # process optimization, cost reduction, and heightened overall efficiency.
        chain_email = prompt_email | self.llm
        res = chain_email.invoke(input={'job_description': str(job), 'link_list': links})
        return res.content


if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))