"""
This program is for choosing the book, creating a database out of it using semantic chunking,
and then sotirng the said database locally. Use this to "teach" the bot new things.
"""
import os
from typing import List
import numpy as np
import tqdm # For visualizing

# This swaps the stdlib sqlite3 with pysqlite3, necessary to amke ChromaDB work
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb as chr # We are using this to save the database

from langchain_community.document_loaders import PyPDFLoader
from tkinter import filedialog as fd # Graphical way of inserting books
# Local LLM via Huggingface
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
# Embedding will happen through a chromaDB database

from langchain_huggingface import HuggingFaceEmbeddings
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


class ChromaEmbeddingFunction:
    """
    A wrapper to make HuggingFace embeddings compatible with ChromaDB's EmbeddingFunction.
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __call__(self, input: List[str]) -> List[np.ndarray]:
        # For ChromaDB the input should be a list of strings and output a list of embeddings.
        return self.embedding_model.embed_documents(input)

# Wrap the huggingface model so that chromaDB can use it
chroma_embed_function = ChromaEmbeddingFunction(embed_model)

def selectBook():

    print("\nWelcome to the libraryMaker!\nPlease press enter and select the book you would like to process.")
    userInput = input()

    while userInput != "":
        print("\nPRESS ENTER!")
        userInput = input()

    filepath = fd.askopenfilename()

    if not filepath.lower().endswith('.pdf') or not os.path.isfile(filepath):
        print("\nError: Please select a valid PDF file.")
        
        return selectBook()

    loader = PyPDFLoader(filepath)
    documents = loader.load()

    fileName = os.path.basename(filepath).split('/')[-1] # Return the name of the file
    return documents, fileName

def semanticChunker(data):

    if not data:
        print("\nError: No content found in the document.")
        return []

    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")

    print("\nSemantic chunking in progress...")

    semantic_chunks = []
    # Visualize the chunking progress with a progress bar
    for i, content in enumerate(tqdm.tqdm([d.page_content for d in data], desc="Chunking Pages")):
        chunk = semantic_chunker.create_documents([content])
        semantic_chunks.extend(chunk)

    # Old method, without progress bar
    #semantic_chunks = semantic_chunker.create_documents([d.page_content for d in data])

    if not semantic_chunks:
        print("\nError: Semantic chunking failed or returned no results.")
        return []

    return semantic_chunks

# Helper function to make sure the same database is not created twice
def check_db_exists(path, name,):
    """
    Check if the Chroma collection already exists.

    Parameters:
    - path (str): The path where the Chroma database is stored.
    - name (str): The name of the collection to check.

    Returns:
    - bool: True if the collection exists, False otherwise.
    """
    chroma_client = chr.PersistentClient(path=path)
    try:
        chroma_client.get_collection(name=name, embedding_function=embed_model)
        return True  # Collection exists
    except Exception:
        return False  # Collection does not exist
    
def create_chroma_db(documents:List):
    """
    Creates a Chroma database using the provided documents, path, and collection name.

    Parameters:
    - documents: An iterable of documents to be added to the Chroma database.
    - path (str): The path where the Chroma database will be stored.
    - name (str): The name of the collection within the Chroma database.

    Returns:
    - Tuple[chromadb.Collection, str]: A tuple containing the created Chroma Collection and its name.
    """
    name = input("\nInsert the name of this database (e.g. the book's name):\n")
    print("\nWhere would you like to store the database?")
    pathEnter = input("\nPress ENTER to choose the location.")
    while pathEnter != "":
        print("\nPRESS ENTER!")
        pathEnter = input()
    
    dataPath = fd.askdirectory(title="Select a folder to store the database")
    chroma_client = chr.PersistentClient(path=dataPath)

    # Check if the collection already exists
    if check_db_exists(dataPath, name):
        print(f"\nThe database '{name}' already exists in the specified path.")
        return None
    
    db = chroma_client.create_collection(name=name, embedding_function=chroma_embed_function)

    # Wrap the document and ID as a list
    for i, d in enumerate(documents):
        #db.add(documents=d, ids=str(i))
        # Alternative way, try if above one does not work
        db.add(documents=[d.page_content], ids=[str(i)])


    return db, name




def main():
    # First, choose the file and process it
    data, file = selectBook()

    # Chunk the processed information
    sem_chunks = semanticChunker(data)

    # Store the chunks in a ChromaDB database
    if sem_chunks:
        create_chroma_db(sem_chunks)
        print(f"\nDatabase has been created for the file {file}!")
    else:
        print("Error: Semantic chunking returned no results. Closing the program.")

if __name__=="__main__":
    main()