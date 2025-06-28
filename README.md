# DriveBot: AI Assistant for Renault Triber (RAG-based Chatbot)

DriveBot is an intelligent chatbot that answers natural language questions from the Renault Triber car manual using a Retrieval-Augmented Generation (RAG) pipeline. It combines semantic search using FAISS with a lightweight HuggingFace transformer model to simulate an AI assistant for vehicle documentation.




## 📁 Project Structure
```
├── app.py                 # Streamlit app
├── vectorstore/           # Prebuilt FAISS vector index
│   ├── index.faiss
│   └── index.pkl
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
```




## 🚀 Demo

🧪 Try it on Streamlit Cloud:  
[Live App Link](https://triberdrivebot.streamlit.app/)



## 🔧 How It Works

### 1. Document Processing (done offline)

- Original PDF manual was converted to plain `.txt`
- Text was split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`
- Embeddings generated using `all-MiniLM-L6-v2`
- Vectors stored in a FAISS index (`vectorstore/`)

### 2. Language Model (LLM)

- Loaded `google/flan-t5-base` using `transformers`
- Wrapped in a `HuggingFacePipeline` for LangChain compatibility
- Answer generated from top-`k` retrieved document chunks

### 3. RetrievalQA Pipeline

- Prompt template guides the model to stay grounded
- Combines semantic retrieval with generation for factual answers

### 4. Web Interface

- Built with Streamlit
- Interactive chat interface with memory (via session state)
- Easy to deploy on Streamlit Cloud
