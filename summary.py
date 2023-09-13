import streamlit as st
from langchain.llms import VertexAI
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def generate_response(txt):
    # Instantiate the LLM model
    llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=1024,
    temperature=0.3,
    top_p=0.8,
    top_k=40,
    verbose=True,
    )
    
    combine_prompt = """
        Seja um editor especialista, seu trabalho é criar resumos conciso do texto a seguir.
        Retorne sua resposta em marcadores que cubram os pontos-chave do texto.
        ```
        {text}
        ```
        BULLET POINT RESUMO:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce', combine_prompt=combine_prompt_template, verbose=True)
    output_summary = chain.run(docs)
    #print(output_summary)
    return output_summary

# Page title
st.set_page_config(page_title='🦜🔗 Summarization App')
st.title('🦜🔗 Summarization App')

# Text input
txt_input = st.text_area('Digite seu texto:', '', height=200)

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Gerar resumo')
    if submitted:
        with st.spinner('Aguarde..'):
            response = generate_response(txt_input)
            result.append(response)

if len(result):
    #st.info(response)
    bullets = response.strip().split('\n')
    for i in bullets:
      st.info(i)
      st.button("Gerar um novo bullet", key=i)