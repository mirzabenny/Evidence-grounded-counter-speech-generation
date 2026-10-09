import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import math


CORPUS_FILE = r"C:\Users\ACER\fact_cn\static_corpus.jsonl"

FAISS_PATH = r"C:\Users\ACER\fact_cn\corpus_improved.index"
TEXT_PATH = r"C:\Users\ACER\fact_cn\texts_improved.npy"



CHUNK_SIZE = 200          # words per chunk (balanced for MiniLM)
CHUNK_OVERLAP = 40        # overlap to preserve context
MIN_CHUNK_WORDS = 40      # skip tiny chunks

BATCH_SIZE = 8000         # safe for your RAM (important!)
TRAIN_SAMPLE = 50000      # sample size for FAISS training




def clean_text(text):
    """
    Remove extra spaces and newlines.
    Makes text consistent before embedding.
    """
    return " ".join(text.split())



def chunk_text(text):
    """
    Split long text into overlapping chunks.
    This improves retrieval precision.
    """
    words = text.split()
    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for i in range(0, len(words), step):

        chunk_words = words[i:i + CHUNK_SIZE]

        # skip small chunks (low information)
        if len(chunk_words) < MIN_CHUNK_WORDS:
            continue

        chunk = " ".join(chunk_words)
        chunks.append(chunk)

    return chunks




def load_corpus():
    """
    Convert each document into multiple chunks.
    Each chunk = Title + Summary + Chunk of Text
    """
    corpus_chunks = []

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:

        for idx, line in enumerate(tqdm(f, desc="Loading corpus")):

            try:
                item = json.loads(line)

                title = clean_text(item.get("title", ""))
                summary = clean_text(item.get("summary", ""))
                text = clean_text(item.get("text", ""))

                if not text:
                    continue

                # base context (global info)
                base_context = f"{title}. {summary}".strip()

                # split text into chunks
                chunks = chunk_text(text)

                for chunk_id, chunk in enumerate(chunks):

                    combined = f"{base_context} {chunk}"

                    corpus_chunks.append({
                        "text": combined,
                        "doc_id": idx,
                        "chunk_id": chunk_id
                    })

            except:
                continue

    return corpus_chunks



def build_corpus():

    data = load_corpus()
    print(f"\nTotal chunks created: {len(data)}")

    texts = [d["text"] for d in data]

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    dim = 384
    nlist = 100

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)

    print("Training FAISS index...")

    sample_texts = texts[:TRAIN_SAMPLE]

    sample_embeddings = model.encode(
        sample_texts,
        batch_size=128
    )

    sample_embeddings = np.array(sample_embeddings).astype("float32")

    faiss.normalize_L2(sample_embeddings)

    index.train(sample_embeddings)

    print("Adding embeddings in batches...")

    num_batches = math.ceil(len(texts) / BATCH_SIZE)

    for i in tqdm(range(num_batches), desc="Indexing batches"):

        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, len(texts))

        batch_texts = texts[start:end]

        batch_embeddings = model.encode(
            batch_texts,
            batch_size=128
        )

        batch_embeddings = np.array(batch_embeddings).astype("float32")

        faiss.normalize_L2(batch_embeddings)

        index.add(batch_embeddings)

    faiss.write_index(index, FAISS_PATH)

    print("FAISS index saved.")

    np.save(TEXT_PATH, np.array(data, dtype=object))

    print("Text metadata saved.")



if __name__ == "__main__":
    build_corpus()