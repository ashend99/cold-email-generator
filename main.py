import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text

# st.markdown(
#     """
#     <style>
#     .stCodeBlock {
#         white-space: pre-wrap; /* Wrap code lines */
#         word-wrap: break-word; /* Break long words if needed */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

def create_streamlit_app(llm, portfolio, clean_text):
    st.title("Cold Main Generator")
    url_input = st.text_input('Enter a URL:', value="https://jobs.nike.com/job/R-36827?from=job%20search%20funnel")
    submit_button = st.button("Submit")

    if submit_button:
        # st.code("Hello Hiring Manager, I am from AtliQ", language='markdown')
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)
            for job in jobs:
                skills = job.get('skills', [''])
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links)
                st.markdown(email)
        except Exception as e:
            st.error(f"An Error Occurred: {e}")

if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)