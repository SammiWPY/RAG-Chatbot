from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import config
from logger import Timer, log_query

def load_vectorstore(persist_dir: str= config.VECTORSTORE_DIR):
    embeddings=HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True}
    )
    return FAISS.load_local(
        persist_dir, embeddings, allow_dangerous_deserialization=True)
    

def build_chain(vectorstore, top_k: int=config.RETRIEVER_TOP_K):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    llm = ChatGroq(api_key=config.GROQ_API_KEY, model=config.LLM_MODEL_NAME, temperature=config.LLM_TEMPERATURE)

    prompt = ChatPromptTemplate.from_messages([
        ("system", config.SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    answer_chain = (
        {"context": lambda values:format_docs(values["documents"]),
         "question": lambda values: values["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    return answer_chain, retriever

def answer_question(question: str, chain, retriever):
    with Timer() as retrieval_timer:
        retrieved_docs = retriever.invoke(question)
    with Timer() as generation_timer:
        answer= chain.invoke({"question":question, "documents": retrieved_docs})
    print(answer)
    log_query(question, answer, len(retrieved_docs), retrieval_timer.elapsed_ms, generation_timer.elapsed_ms)
    return {
        "answer": answer,
        "sources": retrieved_docs,
        "retrieval_latency_ms": retrieval_timer.elapsed_ms,
        "generation_latency_ms": generation_timer.elapsed_ms,
    }


def main():
    vectorstore = load_vectorstore()
    chain, retriever = build_chain(vectorstore)
    question="What do I need to hand in in the final stage"
    answer_question(question, chain, retriever)

if __name__ =="__main__":
    main()