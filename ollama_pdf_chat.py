import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Bölüm 1: PDF Yükleme
loader = PyPDFLoader("agile.pdf")
data = loader.load()

# Bölüm 2: Dokümanı Parçalara Bölme
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# Bölüm 3: Embedding İşlemi
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Bölüm 4: Vektör Veri Tabanı Oluşturma
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="db",
)

# Bölüm 5: Prompt Yapılandırması
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful agile coach. Answer the question using only the context provided from the agile.pdf document. Do not include any external information. If the answer is not covered in the given context, state that you cannot answer based solely on the document."),
    ("user", "Context: {context}\nQuestion: {question}")
])

# Streamlit Uygulaması Başlığı
st.title("Agile Coach: Answer only from agile.pdf")

# Kullanıcıdan Soru Alma
input_text = st.text_input("What's on your mind?")

llm = OllamaLLM(model="llama3.2:1b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    # Vektör araması ile agile.pdf'den ilgili içerikleri çekiyoruz.
    retrieved_docs = vectordb.similarity_search(input_text)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt_data = {"context": context, "question": input_text}
    answer = chain.invoke(prompt_data)
    st.write(answer)
