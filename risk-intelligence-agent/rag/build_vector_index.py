"""Build a local Chroma vector index from risk documents."""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def main():
    base_dir = Path(__file__).resolve().parents[1]
    docs_dir = base_dir / 'data' / 'raw' / 'risk_documents'
    vectorstore_dir = base_dir / 'vectorstore'
    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(docs_dir.glob('*.txt'))
    if not txt_files:
        raise FileNotFoundError(f'No text documents found in {docs_dir}')

    documents = []
    for file_path in txt_files:
        loader = TextLoader(str(file_path), encoding='utf-8')
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(vectorstore_dir),
    )

    print(f'Vector index built at {vectorstore_dir}')


if __name__ == '__main__':
    main()
