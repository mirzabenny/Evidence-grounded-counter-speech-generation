# Evidence-Grounded Counter-Speech Generation (EGCSG)

An evidence-grounded counter-speech generation framework combining semantic retrieval from a static knowledge base, real-time web search, relevance grading, evidence summarization, and evidence-constrained generation with an open-source language model.

## Pipeline

1. Generate two factual search queries from an input hate-speech statement.
2. Retrieve relevant chunks from a FAISS index built from Newsroom documents.
3. Retrieve additional evidence through the Tavily Search API.
4. Grade retrieved documents for relevance.
5. Summarize the selected evidence.
6. Generate respectful counter-speech using the evidence.
7. Evaluate outputs using ROUGE-2, BLEU, BERTScore, Perspective API toxicity scoring, and LLM-based factual-accuracy scoring.

A Streamlit interface is included to demonstrate the pipeline.

## Project structure

```text
fact_cn/
├── app.py
├── updated_cs.py                    # Required by app.py
├── build_corpus_improved.py
├── counter_speech_generation.py
├── extract_combined_subset.py
├── LLM_factual_accuracy.py
├── requirements.txt
├── .gitignore
├── README.md
├── static_corpus.jsonl              # Generated data; normally do not commit
├── corpus_improved.index            # Generated FAISS index; can be rebuilt
├── texts_improved.npy               # Generated metadata; can be rebuilt
├── Original dataset/                 # Check dataset licence before sharing
└── results/                          # Usually exclude from Git
```

Rename your uploaded scripts to the clean filenames shown above if you want this structure. Include `updated_cs.py` if the Streamlit app imports it.

## Requirements

- Python 3.10 or compatible Python 3.x
- [Ollama](https://ollama.com/)
- An Ollama model used by the scripts (`mistral:7b` for generation; `gemma2:2b` for the current factual-scoring script)
- Tavily API key for web retrieval
- Perspective API key for toxicity scoring
- Optional CUDA-capable GPU for faster embedding/model execution

The scripts use packages including `numpy`, `pandas`, `requests`, `faiss`, `sentence-transformers`, `torch`, `tqdm`, `rouge-score`, `bert-score`, `sacrebleu`, and `streamlit`. Create `requirements.txt` with the compatible package versions for your environment, then install:

```bash
pip install -r requirements.txt
```

Install the appropriate FAISS package for your operating system and environment; avoid conflicting CPU and GPU variants.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY>.git
cd <YOUR-REPOSITORY>
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install and start Ollama

Install Ollama from [ollama.com](https://ollama.com/), then pull the required models:

```bash
ollama pull mistral:7b
ollama pull gemma2:2b
```

The scripts call Ollama's local generation endpoint at `http://localhost:11434/api/generate`. Ensure Ollama is running before executing them.

### 4. Configure API keys securely

Do **not** hard-code API keys in source code or commit them to Git. Use environment variables, for example in PowerShell:

```powershell
$env:TAVILY_API_KEY="your-tavily-api-key"
$env:PERSPECTIVE_API_KEY="your-perspective-api-key"
```

Update the scripts to read the values using `os.getenv(...)`. If a key has already been exposed in a source file or shared, revoke/rotate it before publishing.

### 5. Prepare the static corpus

Place the required Newsroom JSONL files where the corpus-preparation script expects them, or update the configured paths. Then run:

```bash
python extract_combined_subset.py
```

This creates `static_corpus.jsonl` when the configured input files are available.

### 6. Build the FAISS index

Check the input and output paths in `build_corpus_improved.py`, then run:

```bash
python build_corpus_improved.py
```

The script chunks documents, creates embeddings using `all-MiniLM-L6-v2`, normalizes the embeddings, trains a FAISS `IndexIVFFlat` index, and saves the index and metadata.

### 7. Run the pipeline or UI

After checking dataset, index, metadata, and output paths:

```bash
python counter_speech_generation.py
```

Run the Streamlit interface with:

```bash
streamlit run app.py
```

`app.py` imports `updated_cs` and calls `process_row()`, so that module must be present and expose the expected function.

### 8. Run factual-accuracy scoring

Configure the input and output paths in `LLM_factual_accuracy.py`, then run:

```bash
python LLM_factual_accuracy.py
```

## Current implementation settings

- Embedding model: `all-MiniLM-L6-v2`
- Chunk size: 200 words
- Chunk overlap: 40 words
- FAISS index: `IndexIVFFlat`
- FAISS `nlist`: 100
- FAISS `nprobe`: 10
- Static retrieval: top 2 chunks per query
- Web retrieval: up to 2 results per query
- Generation model: `mistral:7b`
- Factual-scoring model in the scoring script: `gemma2:2b`

Some scripts use absolute Windows paths. Update these paths or move them to a configuration file/environment variables before running on another machine.

## Evaluation

The code includes or calls:
- **ROUGE-2:** lexical bigram overlap
- **BLEU:** n-gram precision-based comparison
- **BERTScore:** semantic similarity between generated text and a reference
- **Perspective API (`TOXICITY`):** toxicity scoring, subject to API access and availability
- **LLM-based factual-accuracy scoring:** model-based scoring of counter-speech

Results depend on the datasets, references, model settings, retrieved evidence, and external API availability. Refer to the associated research paper for the experimental protocol and reported findings.

## Data and responsible use

- Check dataset licences and terms before redistributing any data. Do not commit datasets unless redistribution is permitted.
- Keep API keys, generated results, large files, local indexes, and model files out of Git unless there is a clear reason to include them.
- Review retrieved evidence and generated responses for relevance, factuality, and potential bias before use.


