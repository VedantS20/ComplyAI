import PyPDF2
from math import ceil
from utils import get_embeddings, get_llm
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import argparse
import datasets
import random
from datasets import Dataset, load_dataset
from format import DatasetConverter, datasetFormats, outputDatasetTypes
from mdc import MDC

N = 15
CHUNK_SIZE = 1024


def create_document_chunks(file_path: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    chunks = []

    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()

    num_chunks = ceil(len(text) / chunk_size)
    embeddings = get_embeddings(True)
    text_splitter = SemanticChunker(embeddings, number_of_chunks=num_chunks)
    chunks = text_splitter.create_documents([text])
    # print(chunks)
    chunks = [chunk.page_content for chunk in chunks]
    return chunks


def load_checkpoint(filename):
    with open(filename, 'r') as f:
        return int(f.read())


def save_checkpoint(state, filename):
    with open(filename, 'w') as f:
        f.write(str(state))


def generate_questions_based_on_context(chunk, no_of_questions_to_generate: int = 5) -> list[str]:
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a synthetic question-answer pair generator. Given a chunk of context about some topic(s), generate {no_of_questions_to_generate} example questions a user could ask and would be answered using information from the chunk. For example, if the given context was a Wikipedia paragraph about the United States, an example question could be 'How many states are in the United States? . Each Question Should start with a number from 1 to {no_of_questions_to_generate}'"),
        ("system", "The questions should be able to be answered in a few words or less. Include only the questions in your response."),
        ("user", "{chunk}")
    ])

    llm = get_llm(isPaid=True)

    # chain = LLMChain(llm=llm,prompt=prompt_template)
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({
        "chunk": chunk,
        "no_of_questions_to_generate": no_of_questions_to_generate
    })

    print(response, "RESPONSE")

    queries = response.split('\n')
    queries = [validate_question(q) for q in queries if validate_question(q)]
    print(queries, "QUestion @@@@@@@@@@@@@@")
    return queries


def validate_question(s: str) -> str:
    if s and s[0].isnumeric:
        return s
    else:
        return ''


def generate_labels(question: str, chunk):

    prompt = """
        Question: {question}\nContext: {context}\n
        Answer this question using the information given in the context above. Here is things to pay attention to: 
        - First provide step-by-step reasoning on how to answer the question. 
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context. 
        - End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct.
        You MUST begin your final answer with the tag "<ANSWER>:".
    """.format(question=question, context=str(chunk))

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful question answerer who can provide an answer given a question and relevant context."),
        ("user", "{prompt}")
    ])

    llm = get_llm(isPaid=True)
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({
        "prompt": prompt
    })

    # print(response, "RES")

    return response


def add_chunk_to_dataset(chunks: list[str], chunk: str, x: int = 5, num_distract: int = 3, p: float = 0.8):
    global ds
    i = chunks.index(chunk)
    qs = generate_questions_based_on_context(
        chunk=chunk, no_of_questions_to_generate=x)
    if qs and len(qs) > 0:
        for q in qs:
            datapt = {
                "id": None,
                "type": None,
                "question": None,
                "context": None,
                "oracle_context": None,
                "cot_answer": None
            }

            datapt["id"] = f"seed_task_{0 if not ds else ds.num_rows}"
            datapt["type"] = "general"
            datapt["question"] = q

            # add num_distract distractor docs
            docs = [chunk]
            indices = list(range(0, len(chunks)))
            indices.remove(i)
            for j in random.sample(indices, num_distract):
                docs.append(chunks[j])
            # decides whether to add oracle document
            oracle = random.uniform(0, 1) < p
            if not oracle:
                docs[0] = chunks[random.sample(indices, 1)[0]]
            random.shuffle(docs)

            d = {
                "title": [],
                "sentences": []
            }

            d["title"].append(["placeholder_title"]*(num_distract+1))
            d["sentences"].append(docs)
            datapt["context"] = d
            datapt["oracle_context"] = chunk

            # add answer to q
            datapt["cot_answer"] = generate_labels(q, chunk)

            # construct model instruction
            context = ""
            for doc in docs:
                context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
            context += q
            datapt["instruction"] = context

            if not ds:
                # init ds
                datapt["id"] = [datapt["id"]]
                datapt["type"] = [datapt["type"]]
                datapt["question"] = [datapt["question"]]
                datapt["context"] = [datapt["context"]]
                datapt["oracle_context"] = [datapt["oracle_context"]]
                datapt["cot_answer"] = [datapt["cot_answer"]]
                datapt["instruction"] = [datapt["instruction"]]
                ds = Dataset.from_dict(datapt)
            else:
                ds = ds.add_item(datapt)


def main():
    OUTPUT_PATH = 'output'
    global ds
    chunks = create_document_chunks(
        file_path='training_documents/Comp_Alcoholic_Beverages(III)_28_08_2023.pdf')
    num_chunks = len(chunks)

    start = 0
    ds = None
    if os.path.exists("checkpoint.txt"):
        start = int(load_checkpoint("checkpoint.txt"))

    for i in range((start//N)*N, len(chunks)):
        chunk = chunks[i]
        save_checkpoint(i, "checkpoint.txt")

        perc = ceil(i / num_chunks * 100)
        with MDC(progress=f"{perc}%"):
            print(f"Adding chunk {i}/{num_chunks}")
            add_chunk_to_dataset(chunks, chunk, 5, 3)

        if (i+1) % N == 0:
            ds.save_to_disk(OUTPUT_PATH + "-checkpoints-" + str(i))
            ds = None

    if ds:
        ds.save_to_disk(OUTPUT_PATH + "-checkpoints-last")

    ds_list = []

    OUTPUT_DIR = './output'

    for filename in os.listdir(os.path.dirname(OUTPUT_DIR)):
        if "-checkpoints-" in filename:
            for f in os.listdir(os.path.dirname(OUTPUT_DIR) + "/" + filename):
                if f.endswith(".arrow"):
                    ds_list.append(Dataset.from_file(
                        os.path.dirname(OUTPUT_DIR) + "/" + filename + "/" + f))

    ds = datasets.concatenate_datasets(ds_list)
    ds.save_to_disk(OUTPUT_DIR)

    format_params = {}
    # Save as .jsonl format
    formatter = DatasetConverter()

    formatter.convert(ds=ds, format='hf', output_path=OUTPUT_DIR,
                      output_type='jsonl', params=format_params)


if __name__ == "__main__":
    with MDC(progress="0%"):
        main()
