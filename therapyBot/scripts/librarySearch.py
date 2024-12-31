"""
This program retrieves relevant passages using a RAG pipeline created for Google
Gemini. A telegram bot uses this program to retrieve factual information.
"""

# Generic tool and library imports
import os
from typing import List
import numpy as np
import re # This is regex library for filtering

# This swaps the stdlib sqlite3 with pysqlite3 (otherwise ChromaDB doesn't work)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb as chr

# LLM and AI imports
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda # Import runnable lambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory 
from langchain_community.chat_message_histories import ChatMessageHistory # This imports chat message history
from langchain_chroma import Chroma  # LangChain's Chroma integration
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini integration
from langchain_core.messages import SystemMessage # Import SystemMessage

# Constants and configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)


class ChromaEmbeddingFunction:
    """
    A wrapper to make HuggingFace embeddings compatible with ChromaDB's EmbeddingFunction.
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __call__(self, input: List[str]) -> List[np.ndarray]:
        # For ChromaDB the input should be a list of strings and output a list of embeddings.
        return self.embedding_model.embed_documents(input)
    
def filteringRetriever(vectorstore, prompt):
    # This function retrieves chunks and filters them

    irrelevant_keywords = ["table of contents", "appendix", "references", "bibliography", 
                           "sources", "index", "scanned", "All Rights Reserved"]
    
    # Retrieve documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    retrieved_docs = retriever.invoke(prompt)

    # Filtering irrelevant documents based on certain keywords using regex
    filtered_docs = [
        doc for doc in retrieved_docs
        if not any(re.search(r'\b' + re.escape(keyword) + r'\b', doc.page_content, re.IGNORECASE) for keyword in irrelevant_keywords)
    ]

    return filtered_docs

def librarySearch(prompt, memory, session_id):
    """
    This function performs the library search and generates a response.
    param prompt: the user prompt
    """
    db_path = "/home/sacuzel/sourceMaterial/bookData/therapyData"
    collection_name = "therapyData"

    # Load the Chroma database
    client = chr.PersistentClient(path=db_path)
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embed_model
    )

    # Retrieve and filter the chunks
    filtered_docs = filteringRetriever(vectorstore, prompt)
    # Check if there are any docs left after filtering (in case all the docs were useless)
    retrieveAttempts = 0 # Keep count on how many times retrieval happened
    while not filtered_docs and retrieveAttempts < 10:
    # In case all docs were useless, try retrieving again
        filtered_docs = filteringRetriever(vectorstore, prompt)
        retrieveAttempts += 1
        print("\nNew data retrieval, attempt # ", retrieveAttempts, "\n")
    
    # Clean up line breaks in the page content of each document before combining them
    cleaned_content = [doc.page_content.replace(r'\n', ' ').replace('\n', ' ') for doc in filtered_docs]
    # Combine filtered docs into a single context string
    context = "".join(cleaned_content)
    context = context.replace(r'\n', ' ') # More cleaning up
    
    # Define the prompt template
    rag_template = """\
    User's Query:\n
    {prompt}

    Context:\n
    {context}

    """

    # We are using the prompt template to create a Langchain prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a licensed therapist. Your name is Walther Cook. \
Your goal is to help the user, who is your client. Therefore gather as much information about the user as you can and be ready to profile him. \
Use the gathered information whenever you think it is necessary and beneficial. \
Use the provided context as your primary source when answering the user query. The context is a relevant passage from a book about psychotherapy. \
If the context does not contain enough information, use your own database to supplement it when aswering. Use real therapy methdos.\
The context is provided by the system, not by the user, so don't explicitly refer to the context during the chat! \
Treat the user with compassion and kindness. Be professional and do not judge. \
Do not let the user know that you are an AI chatbot, instead try to act naturally. \
Start the session by internally checking the chat history. Then move on to introducing yourself and asking for user's name, age, gender and other crucial information. \
Do not write internal processes to the chat. During the chat, it is extremely crucial that you read the chat history and separate user query from the provided context.\n\n"),
        MessagesPlaceholder(variable_name="history"),  # This history placeholder will be filled with chat history
        ("user", rag_template), # The system message is always the first message (no need to clutter the rag prompt)
    ])

    # This is a runnable to print the prompt for debugging
    def print_prompt(input):
        print("\n###############################################################")
        print("\nFull Prompt to LLM:\n")
        print(input , "\n")
        print("\n###############################################################\n")
        return input

    # Setting up the RunnableWithMessageHistory
    conversation_chain = RunnableWithMessageHistory(
        rag_prompt | RunnableLambda(print_prompt) | chat_model, # This defines the chain
        get_session_history= lambda session_id: ChatMessageHistory(messages = memory.chat_memory.messages) , # This takes a session id and chat history
        input_messages_key="prompt",  # The key under which the user message is placed
        history_messages_key="history"  # The key under which the history is placed
    )

    # Here we generate the response
    response = conversation_chain.invoke(
        {"prompt": prompt, "context": context}, {'configurable': {'session_id': session_id}} # Pass in prompt and context
    )

    return response.content # Returning the final result

def main():  # Some testing
    prompt = input("What is on your mind?\n")
    if prompt == "":
        prompt = "What is love?"
    result = librarySearch(prompt)
    print("\nAnswer:", result)


if __name__ == "__main__":
    main()