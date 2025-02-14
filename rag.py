
# for fetching the files from the database

import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load Vector Database
chroma_db_path = "C:/Users/meesw/Desktop/KA/chroma_db"
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings_model)

def get_top_files(query: str, top_k: int = 2):
    results = vector_store.similarity_search(query, k=top_k)
    file_names = list({doc.metadata["file_name"] for doc in results})  # Extract unique file names
    return file_names[:top_k]  # Ensure we return at most `top_k` file names

def main():
    st.title("RAG File Finder")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.form("query_form", clear_on_submit=True):
        user_input = st.text_input("Enter your query:")
        submitted = st.form_submit_button("Search")

        if submitted and user_input:
            top_files = get_top_files(user_input)
            file_response = ", ".join(top_files) if top_files else "No relevant files found."
            st.session_state.chat_history.append(("User", user_input))
            st.session_state.chat_history.append(("Bot", file_response))

    for role, msg in st.session_state.chat_history:
        st.markdown(f"**{role}:** {msg}")

if __name__ == "__main__":
    main()


















# for chatbot and querying the the context



# import os
# import streamlit as st
# import google.generativeai as genai
# from langchain.llms.base import LLM
# from typing import Optional, List, ClassVar
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain.chains import RetrievalQA

# # Initialize Google Gemini
# genai.configure(api_key="Enter_key")

# class GeminiLLM(LLM):
#     model_name: ClassVar[str] = "gemini-1.5-flash"

#     def __init__(self):
#         super().__init__()

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         try:
#             model = genai.GenerativeModel(model_name=self.model_name)
#             response = model.generate_content(prompt)
#             return response.text
#         except Exception as e:
#             return f"Error generating answer: {e}"

#     @property
#     def _identifying_params(self):
#         return {"model_name": self.model_name}

#     @property
#     def _llm_type(self):
#         return "google_gemini"

# # Load Vector Database
# chroma_db_path = "C:/Users/meesw/Desktop/KA/chroma_db"
# embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vector_store = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings_model)

# # Create RetrievalQA chain
# google_llm = GeminiLLM()
# qa_chain = RetrievalQA.from_chain_type(llm=google_llm, chain_type="stuff", retriever=vector_store.as_retriever())

# def main():
#     st.title("RAG Chatbot with History")

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     with st.form("chat_form", clear_on_submit=True):
#         user_input = st.text_input("Your question:")
#         submitted = st.form_submit_button("Send")

#         if submitted and user_input:
#             answer = qa_chain.run(user_input)
#             st.session_state.chat_history.append(("User", user_input))
#             st.session_state.chat_history.append(("Bot", answer))

#     for role, msg in st.session_state.chat_history:
#         st.markdown(f"**{role}:** {msg}")

# if __name__ == "__main__":
#     main()
