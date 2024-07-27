from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv, dotenv_values
from pymongo import MongoClient
import os

load_dotenv()
config = {**os.environ}

client = MongoClient(config['DB_URL'])


def addOneDocumentToKnowledgeBase(documentUrl: str) -> bool:
    collection = client[config['DB_NAME']
                        ][config['VECTOR_EMBEDDINGS_COLLECTION_NAME']]
    doesCollectionExists = collection.find_one({'source': documentUrl})
    if (doesCollectionExists is not None):
        loader = PyPDFLoader(documentUrl)
        docs = loader.load()
        # TODO : Load Only the English Part of the Document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100)
        texts = text_splitter.split_documents(docs)

        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=config['HUGGINGFACEHUB_API_TOKEN'],
            model_name="sentence-transformers/all-MiniLM-l6-v2"
        )
        vector_search = MongoDBAtlasVectorSearch.from_documents(
            documents=texts,
            embedding=embeddings,
            collection=client[config['DB_NAME']
                              ][config['VECTOR_EMBEDDINGS_COLLECTION_NAME']],
            index_name=config['VECTOR_SEARCH_INDEX'],
        )
        return vector_search
    return False


if __name__ == '__main__':
    url = "https://www.fssai.gov.in/upload/notifications/2022/06/62ac3f9dba33cGazette_Notification_Vegan_Food_17_06_2022.pdf"
    print(addOneDocumentToKnowledgeBase(url))
