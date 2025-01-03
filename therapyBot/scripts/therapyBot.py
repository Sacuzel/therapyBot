from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import google.generativeai as genai
import os
from typing import Dict
from langchain.memory import ConversationBufferMemory

from librarySearch import librarySearch

API_KEY = os.environ.get("GEMINI_API_KEY")
TEL_TOKEN = os.environ.get("TELEGRAM_API_KEY")

# Configure the generative model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Dictionary that stores memory for each session
memory_store: Dict[str, ConversationBufferMemory] = {}

async def generate_content(full_prompt: str) -> str:
    try:
        # Pass the user message to the RAG pipeline for processing
        response = await librarySearch(full_prompt)
        #response = model.generate_content(full_prompt)
        return response.text if hasattr(response, 'text') else "Sorry, I couldn't generate a response."
    except Exception as e:
        return f"There was an error generating the response: {str(e)}"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.first_name
    system_message = f"Hello {user_id}! I am an AI based therapist bot. How can I help you today?"
    await update.message.reply_text(system_message)
    

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)  # Get the user id as a string
    user_message = update.message.text

    # Get or create memory for this session
    if user_id not in memory_store:
        memory_store[user_id] = ConversationBufferMemory(return_messages=True)
    memory = memory_store[user_id]

    try:
        # I am passing memory to the prompt (might not be smart if chat history grows, but works so far)
        response_text = librarySearch(user_message, memory, user_id)  # Use the RAG-based generation function

        # Add the prompt to the memory
        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(response_text)

        # We send the response message to the user
        await update.message.reply_text(response_text)


    except Exception as e:
        await update.message.reply_text(f"There was an error processing your request: {str(e)}")

# Build the application
app = ApplicationBuilder().token(TEL_TOKEN).build()

# Add handlers
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

# Run the bot
app.run_polling()


