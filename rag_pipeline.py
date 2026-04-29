import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def build_rag_pipeline(pdf_path: str, api_key: str):
    """
    Builds a RAG pipeline from a PDF file.
    Returns a RetrievalQA chain.
    """
    os.environ["GOOGLE_API_KEY"] = api_key

    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # 3. Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 5. Create LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    # 6. Custom prompt
    prompt_template = """
    You are a helpful AI assistant. Use the following context from the document to answer the question.
    If the answer is not in the context, say "I couldn't find this in the document."

    Context:
    {context}

    Question: {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 7. Build QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain


def ask_question(qa_chain, question: str) -> dict:
    """
    Ask a question using the RAG pipeline.
    Returns answer and source documents.
    """
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:200] for doc in result["source_documents"]]
    }
