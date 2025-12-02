import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools import tool

load_dotenv()

loader = WebBaseLoader(
    web_paths=("https://pt.wikipedia.org/wiki/2025",),
)

docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                            base_url=os.environ.get("BASE_EMBEDDING_URL"),
                            default_headers={"FlowAgent": "Flow Api", "FlowTenant": "cit"})

vector_store = InMemoryVectorStore(embeddings)

document_ids = vector_store.add_documents(documents=all_splits)

@tool(response_format="content_and_artifact")
def get_news(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

SYSTEM_PROMPT = """You are en expert in summarize news in the funny way
You have access to tool to get this year news.

- get_news: use this tool to retrieve embedding news information
"""


def main():
    model = init_chat_model(
        "gpt-4o-mini",
        temperature=0.5,
        timeout=10,
        base_url=os.environ.get("BASE_URL"),
        default_headers={"FlowAgent": "Flow Api"})

    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[get_news],
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "O que aconteceu em fevereiro de 2025?"}]},
    )

    print(response)

if __name__ == "__main__":
    main()