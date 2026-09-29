import os

from dotenv import load_dotenv

load_dotenv()

# ponytail: missing keys are None for now; fail loudly once the code actually uses them
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
DATABASE_URL = os.getenv("DATABASE_URL")
