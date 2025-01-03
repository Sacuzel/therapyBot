import google.generativeai as genai
import os
import pypdf
import re

# This swaps the stdlib sqlite3 with pysqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb as chr
from typing import List

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def load_pdf(file_path):
    """
    Reads the text content from a PDF file and returns it as a single string.

    Parameters:
    - file_path (str): The file path to the PDF file.

    Returns:
    - str: The concatenated text content of all pages in the PDF.
    """
    # Logic to read pdf
    reader = pypdf.PdfReader(file_path)

    # Loop over each page and store it in a variable
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text

def split_text(text: str):
    """
    Splits a text string into a list of non-empty substrings based on the specified pattern.
    The "\n \n" pattern will split the document para by para
    Parameters:
    - text (str): The input text to be split.

    Returns:
    - List[str]: A list containing non-empty substrings obtained by splitting the input text.

    """
    split_text = re.split('\n \n', text)
    return [i for i in split_text if i != ""]

class GeminiEmbeddingFunction(chr.EmbeddingFunction):
    """
    Custom embedding function using the Gemini AI API for document retrieval.

    This class extends the EmbeddingFunction class and implements the __call__ method
    to generate embeddings for a given set of documents using the Gemini AI API.

    Parameters:
    - input (Documents): A collection of documents to be embedded.

    Returns:
    - Embeddings: Embeddings generated for the input documents.
    """
    def __call__(self, input: chr.Documents) -> chr.Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]

def create_chroma_db(documents:List, path:str, name:str):
    """
    Creates a Chroma database using the provided documents, path, and collection name.

    Parameters:
    - documents: An iterable of documents to be added to the Chroma database.
    - path (str): The path where the Chroma database will be stored.
    - name (str): The name of the collection within the Chroma database.

    Returns:
    - Tuple[chromadb.Collection, str]: A tuple containing the created Chroma Collection and its name.
    """
    chroma_client = chr.PersistentClient(path=path)
    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    for i, d in enumerate(documents):
        db.add(documents=d, ids=str(i))

    return db, name

def check_db_exists(path, name):
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
        chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        print("COLLECTION EXISTS!!!")
        return True  # Collection exists
    except Exception:
        return False  # Collection does not exist

def load_chroma_collection(path, name):
    """
    Loads an existing Chroma collection from the specified path with the given name.

    Parameters:
    - path (str): The path where the Chroma database is stored.
    - name (str): The name of the collection within the Chroma database.

    Returns:
    - chromadb.Collection: The loaded Chroma Collection.
    """
    chroma_client = chr.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    return db

def make_rag_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
                Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
                However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
                strike a friendly and converstional tone. \
                If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
    """).format(query=query, relevant_passage=escaped)

    print(f"\nTHE RELEVANT PASSAGE:\n {relevant_passage} \n")
    
    return prompt

def get_relevant_passage(query, db, n_results):
  passage = db.query(query_texts=[query], n_results=n_results)['documents'][0]
  return passage
  
def generate_GEMINI_answer(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

def generate_answer(db,query):
    #retrieve top 3 relevant text chunks
    relevant_text = get_relevant_passage(query,db,n_results=3)
    prompt = make_rag_prompt(query, 
                             relevant_passage="".join(relevant_text)) # joining the relevant chunks to create a single passage
    answer = generate_GEMINI_answer(prompt)

    return answer

def main(userQuery: str):

    if userQuery == None:
        userQuery = "Tell me the exact reference passage you were given."

    # Temporarily disabling this part
    """
    pdf_text = load_pdf(file_path='/home/sacuzel/sourceMaterial/depression.pdf')
    chunked_text = split_text(text=pdf_text)
    if not check_db_exists(path='/home/sacuzel/telegram_t_bot/therapyBot/databases_etc', name="depressionData"):
        print("\nGenerating a new collection!")
        db,name =create_chroma_db(documents=chunked_text, 
                            path='/home/sacuzel/telegram_t_bot/therapyBot/databases_etc', #replace with your path
                            name="depressionData")
        """
    
    print("\nUsing an existing collection")
    db=load_chroma_collection(path='/home/sacuzel/sourceMaterial/bookData/therapyData', name='therapyData')

    print("GENERATING AN API CALL...")
    answer = generate_answer(db,query=userQuery)
    print(answer)

    return answer
    
if __name__=="__main__":
    main(userQuery=None)


