import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

pdf_folder = input("Enter the path to the folder containing the PDF files: ")

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define persistent Chroma storage path
chroma_db_path = "C:/Users/meesw/Desktop/KA/chroma_db"

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"
    return text.strip()

def create_embeddings_for_pdfs(pdf_folder):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            if text:
                chunks = text_splitter.split_text(text)
                documents.extend(
                    Document(page_content=chunk, metadata={"file_name": filename, "chunk_index": i})
                    for i, chunk in enumerate(chunks)
                )

    if documents:
        # Use Chroma.from_documents to enable persistence
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings_model,
            persist_directory=chroma_db_path
        )
        print(f"Vector store created at {chroma_db_path}")
    else:
        print("No valid documents found to process.")

if __name__ == "__main__":
    create_embeddings_for_pdfs(pdf_folder)
