# pip install -r requirements.txt

# @title Import Packages
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# @title Section1: Loading
loader = PyPDFLoader("agile.pdf")
data = loader.load()
data

# @title Section2: Splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splits = text_splitter.split_documents(data)
splits

# @title Section3: Embedding

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# @title Section4: Creating and Storing Vector
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="db",
)

# @title Section5: Querying the Data
question="What is agile development?"
result=vectordb.similarity_search(question)
print(result[0].page_content)

question="How does Agile differ from traditional project management approaches?"
result=vectordb.similarity_search(question)
print(result[0].page_content)

question="What roles are essential in an Agile team, and what are their responsibilities?"
result=vectordb.similarity_search(question)
print(result[0].page_content)

question="How do Agile teams handle changes and adapt to new requirements during a project?"
result=vectordb.similarity_search(question)
print(result[0].page_content)

question="What are some common Agile practices and techniques (e.g., sprints, stand-ups, retrospectives) used to ensure continuous improvement and high-quality outcomes?"
result=vectordb.similarity_search(question)
print(result[0].page_content)