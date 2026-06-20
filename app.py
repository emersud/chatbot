import os
import streamlit as st
from dotenv import load_dotenv

# 1. Carrega as variáveis de ambiente do arquivo .env (apenas para teste local)
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

# 2. Configuração da Página do Streamlit
st.set_page_config(page_title="NVIDIA RAG PDF Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Assistente de Manual em PDF (NVIDIA RAG)")
st.write("Faça perguntas em português sobre o produto com base no manual fornecido.")

# 3. Gerenciamento da API Key da NVIDIA
# Verifica primeiro se a chave está no arquivo .env local ou nas variáveis de sistema
nvidia_api_key = os.environ.get("NVIDIA_API_KEY")

# Se não estiver no .env, tenta buscar nos Secrets do Streamlit (comum no deploy em produção)
if not nvidia_api_key:
    try:
        if "NVIDIA_API_KEY" in st.secrets:
            nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
    except Exception:
        # Ignora falhas se a estrutura de secrets não existir localmente
        pass

if not nvidia_api_key:
    st.info("Por favor, adicione sua NVIDIA_API_KEY no arquivo .env (local) ou nos Secrets do Streamlit (deploy).", icon="🔑")
    st.stop()

# 4. Inicialização e Cache do Banco de Dados RAG (Vetorização do PDF)
@st.cache_resource(show_spinner="Processando o manual em PDF...")
def inicializar_rag():
    nome_arquivo_pdf = "manual.pdf"
    
    if not os.path.exists(nome_arquivo_pdf):
        st.error(f"Arquivo '{nome_arquivo_pdf}' não foi encontrado!")
        st.stop()
    
    loader = PyPDFLoader(nome_arquivo_pdf)
    paginas = loader.load()
    
    # TextSplitter calibrado para os limites da API da NVIDIA (Máx 512 tokens)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,        # Blocos menores evitam estouro de tamanho de tokens
        chunk_overlap=50       # Mantém o contexto fluido entre blocos vizinhos
    )
    docs = text_splitter.split_documents(paginas)
    
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5", 
        nvidia_api_key=nvidia_api_key,
        model_type="passage"
    )
    
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    
    # Retorna o buscador configurado para trazer os 4 blocos mais relevantes
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# Executa a inicialização do RAG
retriever = inicializar_rag()

# 5. Configuração do Modelo de Linguagem (LLM) da NVIDIA (Llama 3.1 8B Instruct)
llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct", 
    nvidia_api_key=nvidia_api_key, 
    temperature=0.2
)

# 6. Definição do Prompt Cross-Lingual (Lê em inglês, responde em português)
template_prompt = """
Você é um assistente técnico especializado e prestativo. 
Os fragmentos de contexto abaixo foram extraídos do manual do produto e ESTÃO EM INGLÊS.
Sua tarefa é analisar o contexto em inglês, mas responder à pergunta do usuário OBRIGATORIAMENTE EM PORTUGUÊS.

Use estritamente as informações fornecidas para responder. Se a resposta não puder ser encontrada no texto, diga explicitamente: "Desculpe, mas essa informação não consta no manual do produto."

Contexto (em inglês):
{context}

Pergunta (em português): {question}
Resposta em português:
"""
prompt = ChatPromptTemplate.from_template(template_prompt)

# Criação do Pipeline RAG (LangChain Expression Language - LCEL)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Histórico do Chat na Interface (Streamlit Session State)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Processei o manual em inglês com sucesso. O que você deseja saber sobre o produto?"}
    ]

# Exibe na tela as mensagens trocadas anteriormente
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 8. Fluxo de Interação: Entrada do Usuário e Resposta da IA
if prompt_usuario := st.chat_input("Ex: Qual é o significado do erro 4?"):
    # Adiciona e exibe a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.write(prompt_usuario)
        
    # Gera e exibe a resposta do assistente baseada no PDF
    with st.chat_message("assistant"):
        with st.spinner("Consultando manual técnico..."):
            try:
                resposta = rag_chain.invoke(prompt_usuario)
                st.write(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
            except Exception as e:
                st.error(f"Erro ao processar a requisição na API da NVIDIA: {e}")