from newsroom import jsonl

# Read entire file:

with jsonl.open("dev.data", gzip=True) as train_file:
    train = train_file.read()

# Read file entry by entry:

with jsonl.open("dev.data", gzip=True) as train_file:
    for entry in train_file:
        print(entry["summary"], entry["text"])
