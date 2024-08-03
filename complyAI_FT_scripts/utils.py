
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.llms import HuggingFaceEndpoint


load_dotenv()
config = {**os.environ}


def get_embeddings(isMistral: bool = False):
    if isMistral:
        api_key = config["MISTRAL_API_KEY"]
        return MistralAIEmbeddings(
            model="mistral-embed",
            api_key=api_key
        )
    else:
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=config['HUGGINGFACEHUB_API_TOKEN'],
            model_name="sentence-transformers/all-MiniLM-l6-v2"
        )


def get_llm(isPaid: bool = False):
    if isPaid:
        api_key = config["MISTRAL_API_KEY"]
        model_name = "mistral-large-latest"
        return ChatMistralAI(api_key=api_key, model_name=model_name)
    else:
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

        return HuggingFaceEndpoint(
            repo_id=repo_id, max_length=500, temperature=0.9, max_new_tokens=500, token=config['HUGGINGFACEHUB_API_TOKEN']
        )
