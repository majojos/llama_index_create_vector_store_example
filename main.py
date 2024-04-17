import os
import faiss
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def main():
    os.environ['HF_HOME'] = '/mnt/big_part/projects/12_llamaindex_from_documents_example/transformers'

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    faiss_index = faiss.IndexFlatL2(384)
    documents = SimpleDirectoryReader("/mnt/big_part/projects/12_llamaindex_from_documents_example/data").load_data(
        show_progress=True)

    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)

    index.storage_context.persist("/mnt/big_part/projects/12_llamaindex_from_documents_example/storage")


if __name__ == '__main__':
    main()
