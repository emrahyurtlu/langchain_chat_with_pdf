import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. List all PDF files in the 'source_files' folder.
pdf_folder = "source_files"  # Folder at the root level
pdf_files = [
    os.path.join(pdf_folder, file)
    for file in os.listdir(pdf_folder)
    if file.lower().endswith(".pdf")
]

# 2. Load each PDF file and extract its content.
documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pdf_data = loader.load()
    # Each PDF file may return multiple document segments, so we extend the list.
    documents.extend(pdf_data)

# 3. Split large documents into manageable chunks.
# This is useful for embedding operations to work on smaller, more coherent text blocks.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splits = text_splitter.split_documents(documents)

# 4. Initialize the HuggingFace model for embeddings.
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs = {'device': 'cpu'},
    encode_kwargs = {'normalize_embeddings': False}
)
##embeddings.model = embeddings.model.to_empty("cpu")

# 5. Create a vector database using Chroma by vectorizing the document chunks.
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="db",  # Directory where the vector database will be stored
)

print("All PDF files have been read and vectorized successfully.")

# Bölüm 5: Prompt Yapılandırması
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful agile coach. Answer the question using only the context provided from the agile.pdf document. Do not include any external information. If the answer is not covered in the given context, state that you cannot answer based solely on the document."),
    ("user", "Context: {context}\nQuestion: {question}")
])

# Streamlit Uygulaması Başlığı
st.title("Agile Coach: Answer only from pdf files in the source_files folder!")

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
