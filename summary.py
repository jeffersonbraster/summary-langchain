from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain.llms import VertexAI
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from typing import List, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Isso permite todas as origens (CORS curinga)
    allow_credentials=True,
    allow_methods=["*"],  # Isso permite todos os métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Isso permite todos os cabeçalhos HTTP
)

class TextInput(BaseModel):
    text: str
    bulletPoint: Optional[str] = Field(default=None)
    numBulletPoints: Optional[int] = Field(default=None)
    allBulletPoints: Optional[List[str]] = Field(default=None)

class BulletPoint(BaseModel):
    bullet: str

# Instantiate the LLM model
llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=1024,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

def generate_response(txt: str, numBulletPoints: int) -> List[BulletPoint]:
    combine_prompt = f"""
        Como um editor especializado, sua tarefa é resumir o texto a seguir em tópicos concisos, utilizando a língua portuguesa.
        Por favor, retorne sua resposta em bullets points que abordem os principais pontos do texto.
        Quantidade de bullets points solicitados: {numBulletPoints}
        
        --- Início do Texto ---
        {{text}}
        --- Fim do Texto ---
        
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
    
    # Convert the output to a list of bullet points
    bullets = output_summary.strip().split('\n')
    return [BulletPoint(bullet=bullet) for bullet in bullets]

def generate_new_bullet_response(txt: str, bulletPoint: str, allBulletPoints: List[str]) -> List[BulletPoint]:
    all_bullet_points_str = ', '.join([str(item) for item in allBulletPoints])
    combine_prompt = f"""
        Como um editor especializado, sua tarefa é gerar novos bullets de resumos concisos do texto a seguir.
        
        Refaça o bullet point a seguir, levando em consideração o contexto do texto e não faça duplicações com os outros bullets já gerados:
        Bullet point a ser refeito:
        {bulletPoint}
        
        Bullets points já gerados:
        {all_bullet_points_str}
        
        Texto original:
        {{text}}
                
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
    
    # Convert the output to a list of bullet points
    bullets = output_summary.strip().split('\n')
    # Remove duplicates
    unique_bullets = []
    for bullet in bullets:
        if bullet not in allBulletPoints and bullet not in unique_bullets:
            unique_bullets.append(bullet)
    return [BulletPoint(bullet=bullet) for bullet in unique_bullets]

@app.post("/generate_summary/", response_model=List[BulletPoint])
async def generate_summary(text_input: TextInput):
    response = generate_response(text_input.text, text_input.numBulletPoints)
    return response

@app.post("/generate_new_summary/", response_model=List[BulletPoint])
async def generate_new_bullet(text_input: TextInput):
    response = generate_new_bullet_response(text_input.text, text_input.bulletPoint, text_input.allBulletPoints)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
