import csv


# this function is used to create dataset from kaggle NER corpus
# kaggle dataset path: https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data
# we took only the words with pos value = "NN"
def save_dataset_from_kaggle_csv(kaggle_path='ner.csv', output_path='dataset.txt'):
    all_words = []
    all_lines = []
    with open(kaggle_path) as ner_file:
        our_data = csv.reader(ner_file)
        for row in our_data:
            neu_line = []
            for word in row:
                neu_line.append(word)
            all_lines.append(neu_line)
    for i, line in enumerate(all_lines):
        try:
            pos = line[10]
            if pos == "NN":
                text = all_lines[i - 1][9]
                if text == '__END1__':
                    continue
                all_words.append(text)
        except IndexError:
            pass

    with open(output_path, 'w') as write_file:
        for word in all_words:
            write_file.write(word + '\n')
