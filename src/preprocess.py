
## use OS to read all files in the folder output is a list
## run through each file name: 
### send to ingest, using pymupdfloader, use alazyloader to read file
### send to chunk function extract chunks with overlap of 300
## merge all chunks of document into a single list
## proceed to embedding phase
## save all into FAISS vector store

### Resources:  https://stackabuse.com/python-async-await-tutorial/


import os
import asyncio
import itertools
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document


FILE_PATH = "./data/"
VECTORSTORE = "./vectorstore/agriquery_faiss_index"

# initialise the recursive method
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

# ingest data function. We use async here to allow for asynchronuous/continual processing 
async def ingest(file_name, path):
    """
    loads content of file using pymupdf
    input (str): file names and file path
    output (list): file content divided by pages
    """
    pages = []
    loader = PyMuPDFLoader(path + file_name)

    async for page in loader.alazy_load():
        pages.append(page)

    return pages


# chunk function
def chunk(file_content):
    """
    chunks content of file using langchain recursive splitter
    input (list): file content divided by pages
    output (list): chunks with overlaps defined
    """
    chunks = []

    for page in file_content:
        docs = [Document(page_content=page.page_content)]
        texts = splitter.split_documents(docs)
        chunks.append(texts)

    return list(itertools.chain(*chunks))


# embed funtion and store to FAISS store
def embed(chunks):
    """
    embed the chunks using hugging face sentence transformer
    input (list): chunks
    output (list): list of vectors
    """

    embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTORSTORE)


# main function
async def main():
    """
    main function that runs as file is called
    """
    total_chunks = []
    files = os.listdir(FILE_PATH)
    task = [ingest(file_name, FILE_PATH) for file_name in files]
    page_list = await asyncio.gather(*task) 
    for pages in page_list:
        # pages = await ingest(file_name, FILE_PATH)
        # print(f"Total length of pages for {file_name} is: {len(pages)}")
        # call the chunk function
        chunks = chunk(pages)
        # print(len(chunks))
        total_chunks.append(chunks)

    # flatten the list of lists, make it suitable for embedding
    chunks = list(itertools.chain(*total_chunks))
    print(f"Total length of chunks is: {len(chunks)}")
    embed(chunks)

    print("Success!")


if __name__ == "__main__":
    asyncio.run(main())