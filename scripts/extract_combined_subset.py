import json
import random
#Dataset paths
TRAIN = r"C:\Users\ACER\fact_cn\release\train.jsonl\train-stats.jsonl"
DEV   = r"C:\Users\ACER\fact_cn\release\dev.jsonl\dev-stats.jsonl"
TEST  = r"C:\Users\ACER\fact_cn\release\test.jsonl\test-stats.jsonl"

OUTPUT = r"C:\Users\ACER\fact_cn\static_corpus.jsonl"
#Loads documents from a JSONL file
def load_with_limit(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line) #Parses the JSON string in 'line' and converts it into a Python dictionary. Before:'{"title":"History"}', After : {"title":"History"}
                items.append({ 
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "text": entry.get("text", "")
                })
            except:
                continue
    return items

corpus = []

corpus += load_with_limit(DEV)

corpus += load_with_limit(TRAIN)

corpus += load_with_limit(TEST)

random.shuffle(corpus)

with open(OUTPUT, "w", encoding="utf-8") as f:
    for item in corpus:
        f.write(json.dumps(item) + "\n")