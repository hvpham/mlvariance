import json
import argparse
import os
import pickle

from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from collections import Counter

from sklearn.model_selection import train_test_split

# tokenizers
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()


def get_clean_text(text):
    if isinstance(text, float):
        return ''

    clean_text = text.lower().replace('<br />', '\n').replace('<br>', '\n').replace('\\n', '\n').replace('&#xd;', '\n')
    return clean_text


def tokenization_text(data, sentence_limit, word_limit):
    docs = []
    word_counter = Counter()
    for text in data:

        sentences = list()

        for paragraph in get_clean_text(text).splitlines():
            sentences.extend([s for s in sent_tokenizer.tokenize(paragraph)])

        words = list()
        for s in sentences[:sentence_limit]:
            w = word_tokenizer.tokenize(s)[:word_limit]
            # if sentence is empty (due to removing punctuation, digits, etc.)
            if len(w) == 0:
                continue
            words.append(w)
            word_counter.update(w)

        # if all sentences were empty
        if len(words) == 0:
            continue

        docs.append(words)

    return docs, word_counter


def encode_and_pad(input_docs, word_map, sentence_limit, word_limit):
    encoded_docs = list(
        map(lambda doc: list(
            map(lambda s: list(
                map(lambda w: word_map.get(w, word_map['<unk>']), s)
            ) + [0] * (word_limit - len(s)), doc)
        ) + [[0] * word_limit] * (sentence_limit - len(doc)), input_docs)
    )
    sentences_per_document = list(map(lambda doc: len(doc), input_docs))
    words_per_sentence = list(
        map(lambda doc: list(
            map(lambda s: len(s), doc)
        ) + [0] * (sentence_limit - len(doc)), input_docs)
    )
    return encoded_docs, sentences_per_document, words_per_sentence


def preprocess_train(data, word_counter, min_word_count, sentence_limit, word_limit):
    # create word map
    word_map = dict()
    word_map['<pad>'] = 0
    for word, count in word_counter.items():
        if count >= min_word_count:
            word_map[word] = len(word_map)
    word_map['<unk>'] = len(word_map)
    #print('\nTraining data: discarding words with counts less than %d, the size of the vocabulary is %d.\n' % (min_word_count, len(word_map)))

    # encode and pad
    #print('Training data: encoding and padding...\n')
    encoded_train_docs, sentences_per_train_document, words_per_train_sentence = \
        encode_and_pad(data, word_map, sentence_limit, word_limit)

    return encoded_train_docs, sentences_per_train_document, words_per_train_sentence, word_map


def preprocess_test(data, word_map, sentence_limit, word_limit):
    # encode and pad
    #print('\nTest data: encoding and padding...\n')
    encoded_test_docs, sentences_per_test_document, words_per_test_sentence = \
        encode_and_pad(data, word_map, sentence_limit, word_limit)

    return encoded_test_docs, sentences_per_test_document, words_per_test_sentence


def get_cats_ids(data_folder, base_folder, selected_cats):
    json_file = os.path.join(data_folder, base_folder, 'meta_Software.json')

    pros_meta = []
    with open(json_file, 'r') as f:
        for pro_line in f:
            pro_meta = json.loads(pro_line)
            pros_meta.append(pro_meta)

    #cats_count = dict()
    ids_cats_map = dict()
    ids_subcats_map = dict()

    def normalize(cat_list):
        cat_list = cat_list[:3]
        cat_list = [cat.replace('&amp;', '&') for cat in cat_list]
        return cat_list

    for pro_meta in pros_meta:
        cat_list = pro_meta['category']
        id = pro_meta['asin']
        if len(cat_list) == 3:
            cat_list = normalize(cat_list)
            cat2 = cat_list[1]
            cat3 = cat_list[2]

            selected_subcats = selected_cats.get(cat2, None)
            if selected_subcats is not None and cat3 in selected_subcats:
                ids_cats_map[id] = cat2
                ids_subcats_map[id] = cat3
    return ids_cats_map, ids_subcats_map


def get_selected_cats(data_folder, base_folder):
    json_file = os.path.join(data_folder, base_folder, 'selected_cat.json')
    with open(json_file, 'r') as f:
        cats_json = json.load(f)

    cats = dict()

    for cat_json in cats_json:
        cat = cat_json['cat']
        subcats = cat_json['subcats']

        cats[cat] = subcats

    return cats


def get_reviews(data_folder, base_folder, ids_cats_map, ids_subcats_map):
    json_file = os.path.join(data_folder, base_folder, 'Software.json')

    review_texts = []
    review_cats = []
    review_sub_cats = []

    with open(json_file, 'r') as f:
        for line in f:
            review = json.loads(line)
            id = review['asin']
            cat = ids_cats_map.get(id, None)
            if cat is not None:
                review_text = review.get('reviewText', None)
                if review_text is not None:
                    review_texts.append(review_text)
                    review_cats.append(cat)
                    review_sub_cats.append(ids_subcats_map[id])

    return review_texts, review_cats, review_sub_cats


def coarse_filter(data, coarse_targets, fine_targets, filter_list):
    filter_idxs = [i for i, coarse_target in enumerate(coarse_targets) if coarse_target in filter_list]

    new_data = [data[i] for i in filter_idxs]
    new_coarse_targets = [coarse_targets[i] for i in filter_idxs]
    new_fine_targets = [fine_targets[i] for i in filter_idxs]
    return new_data, new_coarse_targets, new_fine_targets


def fine_filter(data, coarse_targets, fine_targets, filter_list, ratio):
    ids = list(range(len(fine_targets)))

    filter_idxs = [i for i, fine_target in enumerate(fine_targets) if fine_target in filter_list]

    exclude_inds = filter_idxs[:int(ratio*len(filter_idxs)/10)]
    include_inds = [i for i in ids if i not in exclude_inds]

    new_data = [data[i] for i in include_inds]
    new_coarse_targets = [coarse_targets[i] for i in include_inds]
    new_fine_targets = [fine_targets[i] for i in include_inds]

    return new_data, new_coarse_targets, new_fine_targets, include_inds


def map_targets(coarse_targets, coarse_classes):
    class_to_idx = {_class: i for i, _class in enumerate(coarse_classes)}
    new_targets = [class_to_idx[t] for t in coarse_targets]

    return new_targets


def holdout_indexs(targets, holdout_class):
    holdout = [(t in holdout_class) for t in targets]

    return holdout


def store_data(data, target, holdout, ids, root, base_folder, file_name):

    (encoded_docs, sentences_per_document, words_per_sentence) = data

    entry = {}
    entry['data'] = {
        'docs': encoded_docs,
        'sentences_per_document': sentences_per_document,
        'words_per_sentence': words_per_sentence}
    entry['labels'] = target
    entry['holdout'] = holdout
    entry['ids'] = ids

    folder_path = os.path.join(root, base_folder)
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(root, base_folder, file_name)

    with open(file_path, 'wb') as f:
        pickle.dump(entry, f)


def store_word_map(word_map, root, base_folder):
    folder_path = os.path.join(root, base_folder)
    os.makedirs(folder_path, exist_ok=True)

    with open(os.path.join(folder_path, 'word_map.json'), 'w') as j:
        json.dump(word_map, j)


parser = argparse.ArgumentParser(description='Prepare the Amazon Review holdout dataset')
parser.add_argument('data_folder', help='data folder path')
parser.add_argument('mode', choices=['holdout'], help='data prepare mode')

args = parser.parse_args()

data_folder = args.data_folder
mode = args.mode

base_folder = 'amazon/software'

NEWDATA_BASE_FOLDER = 'amazon/'+mode+'_%s'

selected_cats = get_selected_cats(data_folder, base_folder)
ids_cats_map, ids_subcats_map = get_cats_ids(data_folder, base_folder, selected_cats)

review_texts, review_cats, review_sub_cats = get_reviews(data_folder, base_folder, ids_cats_map, ids_subcats_map)

NEW_CLASSES = ["Education & Reference",
               "Business & Office",
               "Children's",
               "Utilities",
               "Design & Illustration",
               "Accounting & Finance",
               "Video",
               "Music",
               "Programming & Web Development",
               "Networking & Servers"]

review_texts, review_cats, review_sub_cats = coarse_filter(review_texts, review_cats, review_sub_cats, NEW_CLASSES)

# create train test split:
train_data, test_data, train_cats, test_cats, train_sub_cats, test_sub_cats =\
    train_test_split(review_texts, review_cats, review_sub_cats, test_size=0.1, random_state=69)

train_data, val_data, train_cats, val_cats, train_sub_cats, val_sub_cats =\
    train_test_split(train_data, train_cats, train_sub_cats, test_size=0.1, random_state=69)

Education_Reference = ["Languages", "Maps & Atlases", "Test Preparation", "Dictionaries", "Religion"]  # 0
Business_Office = ["Office Suites", "Document Management", "Training", "Word Processing", "Contact Management"]  # 1
Children_s = ["Early Learning", "Games", "Math", "Reading & Language", "Art & Creativity"]  # 2
Utilities = ["Backup", "PC Maintenance", "Drivers & Driver Recovery", "Internet Utilities", "Screen Savers"]  # 3
Design_Illustration = ["Animation & 3D", "Training", "CAD", "Illustration"]  # 4
Accounting_Finance = ["Business Accounting", "Personal Finance", "Check Printing", "Point of Sale", "Payroll"]  # 5
Video = ["Video Editing", "DVD Viewing & Burning", "Compositing & Effects", "Encoding"]  # 6
Music = ["Instrument Instruction", "Music Notation", "CD Burning & Labeling", "MP3 Editing & Effects"]  # 7
Programming_Web_Development = ["Training & Tutorials", "Programming Languages", "Database", "Development Utilities", "Web Design"]  # 8
Networking_Servers = ["Security", "Firewalls", "Servers", "Network Management", "Virtual Private Networks"]  # 9

holdout_class_list = []
holdout_class_list.extend(Education_Reference)
holdout_class_list.extend(Business_Office)
holdout_class_list.extend(Children_s)
holdout_class_list.extend(Utilities)
holdout_class_list.extend(Design_Illustration)
holdout_class_list.extend(Accounting_Finance)
holdout_class_list.extend(Video)
holdout_class_list.extend(Music)
holdout_class_list.extend(Programming_Web_Development)
holdout_class_list.extend(Networking_Servers)

#holdout_class_list = ["Languages"]
#holdout_class_list = Children_s


sentence_limit = 15
word_limit = 20
min_word_count = 5

train_data, word_counter = tokenization_text(train_data, sentence_limit, word_limit)
val_data, word_counter = tokenization_text(val_data, sentence_limit, word_limit)
test_data, word_counter = tokenization_text(test_data, sentence_limit, word_limit)

for a in range(11):
    # for a in [0, 1, 2, 5, 10]:
    # for a in [10]:
    for holdout_class in holdout_class_list:
        if type(holdout_class) is list:
            holdout_class_name = '-'.join(holdout_class.replace(' ', '_'))
        else:
            holdout_class_name = holdout_class.replace(' ', '_')
            holdout_class = [holdout_class]

        new_base_folder = NEWDATA_BASE_FOLDER % (("%s_%d") % (holdout_class_name, a))

        print("Prepare data for holdout: %s with ratio: %d and saving to: %s" % (str(holdout_class), a, new_base_folder))

        new_train_data, new_train_cats, new_train_sub_cats, new_train_ids = \
            fine_filter(train_data, train_cats, train_sub_cats, holdout_class, a)

        new_val_data, new_val_cats, new_val_sub_cats, new_val_ids = \
            fine_filter(val_data, val_cats, val_sub_cats, holdout_class, a)

        new_train_targets = map_targets(new_train_cats, NEW_CLASSES)
        new_train_holdout_labels = holdout_indexs(new_train_sub_cats, holdout_class)

        new_val_targets = map_targets(new_val_cats, NEW_CLASSES)
        new_val_holdout_labels = holdout_indexs(new_val_sub_cats, holdout_class)

        new_test_targets = map_targets(test_cats, NEW_CLASSES)
        new_test_holdout_labels = holdout_indexs(test_sub_cats, holdout_class)
        new_test_ids = list(range(len(new_test_holdout_labels)))

        encoded_train_docs, sentences_per_train_document, words_per_train_sentence, word_map = \
            preprocess_train(new_train_data, word_counter, min_word_count, sentence_limit, word_limit)

        encoded_val_docs, sentences_per_val_document, words_per_val_sentence = \
            preprocess_test(new_val_data, word_map, sentence_limit, word_limit)

        encoded_test_docs, sentences_per_test_document, words_per_test_sentence = \
            preprocess_test(test_data, word_map, sentence_limit, word_limit)

        store_word_map(word_map, data_folder, new_base_folder)

        store_data((encoded_train_docs, sentences_per_train_document, words_per_train_sentence), new_train_targets,
                   new_train_holdout_labels, new_train_ids, data_folder, new_base_folder, 'train_batch')
        store_data((encoded_val_docs, sentences_per_val_document, words_per_val_sentence), new_val_targets,
                   new_val_holdout_labels, new_val_ids, data_folder, new_base_folder, 'val_batch')
        store_data((encoded_test_docs, sentences_per_test_document, words_per_test_sentence), new_test_targets,
                   new_test_holdout_labels, new_test_ids, data_folder, new_base_folder, 'test_batch')
