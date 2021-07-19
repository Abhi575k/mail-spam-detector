# collect words from the dataset
def collect_words(data, label):
    collected_words = " "
    for val in data.text[data["label"] == label]:
        val = str(val)
        tokens = val.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        for words in tokens:
            collected_words = collected_words + words + " "
    return collected_words