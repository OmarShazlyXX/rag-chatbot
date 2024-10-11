from consts import INDEX_NAME
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama.llms import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from typing import Any, Dict, List
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables import RunnablePassthrough


def run_llm(query: str, chat_history):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    rag_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7, "k": 5}
    )
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    chat = OllamaLLM(model="llama3.1:8b")
    retrieval_chain = create_stuff_documents_chain(chat, retrieval_prompt)
    history_aware_retriever = create_history_aware_retriever(
    chat, rag_retriever, rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=retrieval_chain
    )

   
    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result
   
    

if __name__ == "__main__":
    ans = run_llm(query= "what is a langchain chain ?", chat_history= [])
    print(ans)
    
    