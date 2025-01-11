A simple chat with pdf application
(Langhchain, RAG, Qdrant, Mistral(Groq))

Get Qdrant runnning using Docker: 
docker run -p 6333:6333 -v .:/qdrant/storage/ qdrant/qdrant

Run ingest_bot.py to set up Qdrant DB
then 
streamlit run app_bot.py