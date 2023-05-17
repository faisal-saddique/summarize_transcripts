import streamlit as st
import os
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide")
st.title("Summarize transcripts")

openai_api_key = os.environ["OPENAI_API_KEY"]

llm = OpenAI(model='text-davinci-003',temperature=.3, openai_api_key=openai_api_key)

accepted_file_extensions = ['.txt']


path = st.file_uploader("Choose a text or document file",
                        accept_multiple_files=True, type=accepted_file_extensions)


text_input = st.text_area("Please paste the transcription text: ", height=400)

uploaded_files = []

if path or text_input:
    if path:
        for uploaded_file in path:
            # Display file name
            st.write(f"Reading file {uploaded_file.name}")

            # Display file contents
            if uploaded_file.name.endswith('.txt'):
                contents = uploaded_file.read().decode("utf-8")

                doc = Document(page_content=contents)

                uploaded_files.append(doc)

                # st.write(doc)

            else:
                st.warning(
                    f"File {uploaded_file.name} has an unsupported file extension.")

    if text_input:
        doc = Document(page_content=text_input)
        uploaded_files.append(doc)

    chunk_size = st.number_input("Please enter the chunk size you want to keep for text splitting:",min_value=500)

    # st.write(uploaded_files)

    if chunk_size:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=5,
            # separators=["---"]
            )

        docs = text_splitter.create_documents([item.page_content for item in uploaded_files])
        
        # docs = docs[40:50]

        st.success(f"Now we have {len(docs)} documents and the first one has {llm.get_num_tokens(docs[0].page_content)} tokens")

        # st.write(docs)

        key_parameters = st.text_area("Please enter the key parameters that you want to consider while summarizing:",height=300)

        if key_parameters:
            parameters = key_parameters.split('\n')
                
            map_prompt = """
            Write a summary in markdown of the following text enclosed in triple backticks (```) by strictly including these sections in it:
            """        
            for param in parameters:
                map_prompt += f"\n{param}"
                    
            map_prompt += """\n\n```{text}```\n\nSUMMARY:"""
            

            st.write("Updated prompt looks like:")
            st.text(map_prompt)

            # map_prompt = """
            # Write a concise summary of the following:
            # "{text}"
            # CONCISE SUMMARY:
            # """

            map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

            combine_prompt = """
            Write a summary in bullet points of the following text enclosed in triple backticks (```) by making bullets of these parameters:
            """
            for param in parameters:
                combine_prompt += f"\n{param}"

            combine_prompt += """\n\n```{text}```\n\nBULLET POINT SUMMARY:"""

            combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

            summary_chain = load_summarize_chain(llm=llm,
                                                chain_type='map_reduce',
                                                map_prompt=map_prompt_template,
                                                combine_prompt=combine_prompt_template,
                                                verbose=True
                                                )
            
            if st.button("Summarize"):
                output = summary_chain.run(docs)

                st.success(f"Summary using map reduce method is: {output}")