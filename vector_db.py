import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

scripts = {
   'lion_king':'THE_LION_KING.pdf',
   'into_the_spiderverse':'SPIDER-MAN_INTO_THE_SPIDER-VERSE.pdf' 
}

def prep_manuscripts(scripts, separators = ["\n\n", "\n", " ", ""], size=550, overlap=50):
    manuscripts =[]
    manuscripts_path = os.path.join(os.getcwd(),'raw_manuscripts')
    
    for pdf_key in scripts.keys():
        py_pdf_loader_object = PyPDFLoader(os.path.join(manuscripts_path, scripts[pdf_key]))
        manuscripts.extend(py_pdf_loader_object.load())

    manuscript_chunker = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separators= separators)
    chunked_manuscripts = manuscript_chunker.split_documents(manuscripts)
    return chunked_manuscripts

def create_vector_store__sub(chunked_data):
    chromadb_dir = os.path.join(os.getcwd(),'chroma_db','movie_manuscripts')
    embedding_model = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(documents=chunked_data, embedding=embedding_model, persist_directory=chromadb_dir)
    return vector_store

def check_if_vector_store_exists():
    chromadb_dir = os.path.join(os.getcwd(),'chroma_db','movie_manuscripts')
    embedding_model = OpenAIEmbeddings()
    vector_store = Chroma(persist_directory=chromadb_dir, embedding_function=embedding_model)
    if vector_store._collection.count() > 0:
        return vector_store
    else:
        return  None

def create_vector_store():
    if check_if_vector_store_exists():
        print("Vector store already exists")
        print("Loading pre-existing vector store into memory...")
        return check_if_vector_store_exists()
    else:
        print("Vector store not yet created")
        print("Creating vector store....")
        chunked_data = prep_manuscripts()
        vector_store = create_vector_store__sub(chunked_data)
        print("Vector store created")
        return vector_store