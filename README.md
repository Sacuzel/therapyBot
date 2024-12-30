TherapyBot: A Generative Telegram Bot Powered by Books

This project features a Telegram bot designed to offer professional advice in any chosen field, leveraging generative AI and a custom-built knowledge database sourced from university textbooks and other relevant books. The bot processes user queries using Retrieval-Augmented Generation (RAG), providing accurate and detailed responses based on the data it has been trained on.

Key Components
TherapyBot
The Telegram bot that interacts with users and generates contextually relevant responses. The bot uses Google Gemini's API for natural language processing and integrates with a locally stored database.

LibraryMaker
A tool to upload books, break them into semantically meaningful chunks, and store them in a local Chroma database for later retrieval. This component 'teaches' the bot new information from the books you feed it.
With larger books and weaker CPU's chunking can take a very long time. The provided book should be in a pdf format with selectable text (no OCR is implemented). Currently the bot can use one book at a time.

librarySearch
A retrieval pipeline that queries the Chroma database using the RAG technique. It fetches relevant excerpts from the stored books, enabling the bot to answer user questions based on the provided context.

How It Works

Add a Book:
Use the LibraryMaker to select a PDF book and break it into semantic chunks that will be stored in a Chroma database. This is the core process for teaching the bot new knowledge.

Train the Bot:
The librarySearch program retrieves relevant information from the Chroma database to answer user queries. When a user interacts with the bot, it uses this data to generate a precise response.

Chat with the Bot:
The TherapyBot interacts with users, fetching answers to their queries based on the database and returning detailed, context-aware responses. The bot leverages the generative capabilities of the Gemini API combined with locally stored data.

Setup
1. Clone the repository:

git clone https://github.com/Sacuzel/therapyBot.git

2. Install dependencies:

pip install -r requirements.txt

3. Set up environment variables for API keys:

GEMINI_API_KEY: Your Google Gemini API key.
TELEGRAM_API_KEY: Your Telegram bot API token.

4. Run the programs:

LibraryMaker: Upload books and build a database. Only use this if you have new books that you want to process.
TherapyBot: Start interacting with the bot.
librarySearch: Queries the knowledge base and processes responses. Is automatically called by TherapyBot.

Use Case
This project is ideal for those looking to create a highly specialized chatbot that provides expert advice in a particular field, such as therapy, engineering, or programming. By feeding the bot academic textbooks, you can make it proficient in specific areas, enhancing its responses and usability. At the moment the bot's capabilities are limited, but this is a working framework for further development.
