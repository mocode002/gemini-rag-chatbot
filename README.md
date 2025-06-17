# Gemini RAG Chatbot

Upload PDF, CSV, or TXT documents and chat with them using Google's Gemini 2.0 Flash model.

## 🚀 How to Run

1. Create a `.env` file with your Gemini API key:
    ```
    GOOGLE_API_KEY=your_google_api_key_here
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # on Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run rag_bot.py
    ```

## 🛠 Built With
- [LangChain](https://www.langchain.com/)
- [Gemini 2.0 Flash (Google Generative AI)](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
