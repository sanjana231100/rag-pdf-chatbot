from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from src.hybrid_retriever import hybrid_search, build_bm25_index
from src.reranker import rerank
from dotenv import load_dotenv
from typing import List
import os

load_dotenv()

LLM_MODEL = "llama-3.3-70b-versatile"

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


class HybridRetriever(BaseRetriever):
    """
    LangChain-compatible retriever that wraps our hybrid search.
    This lets us plug hybrid search into LangChain chains
    the same way as a standard vectorstore retriever.
    """

    vectorstore: object
    bm25_index: object
    chunks: List[Document]
    top_n: int = 20

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        candidates = hybrid_search(
            self.vectorstore,
            self.bm25_index,
            self.chunks,
            query,
            top_n=20
        )
        reranked = rerank(query, candidates, top_k=5)
        return reranked


def build_rag_chain(vectorstore, chunks):
    """
    Build a conversational RAG chain using hybrid retrieval.

    Args:
        vectorstore : Chroma vectorstore object from vectorstore.py
        chunks      : list of Document objects (needed for BM25 index)

    Returns:
        a RunnableWithMessageHistory chain ready to answer questions
    """

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    bm25_index = build_bm25_index(chunks)

    retriever = HybridRetriever(
        vectorstore=vectorstore,
        bm25_index=bm25_index,
        chunks=chunks,
        top_n=20
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the latest user question, "
                   "reformulate the question to be standalone and clear. "
                   "Do NOT answer it, just reformulate if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions strictly \
based on the provided document context.

Rules:
- Only use information from the context below to answer
- If the answer is not in the context, say "I could not find that information in the document"
- Be concise and clear

Context:
{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_chain


def ask(chain, question, session_id="default"):
    """
    Ask a question and get back the answer + source documents.

    Args:
        chain      : the chain from build_rag_chain()
        question   : the user's question as a plain string
        session_id : unique ID per conversation

    Returns:
        dict with keys:
            "answer"           : the LLM's answer string
            "source_documents" : list of Document objects used to answer
    """

    response = chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )

    return {
        "answer": response["answer"],
        "source_documents": response["context"]
    }