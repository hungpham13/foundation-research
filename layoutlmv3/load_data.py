from datasets import Dataset, DatasetDict, Sequence, Features, ClassLabel, Value, Image
import json, os
from pathlib import Path

def split_train_test(data, ratio):
    import random
    # Shuffle the data to ensure randomness
    random.shuffle(data)
    
    # Calculate the split index
    train_size = int(len(data) * ratio)
    
    # Split the data
    train_set = data[:train_size]
    test_set = data[train_size:]
    
    return train_set, test_set

def preprocess_label(label):
    if label == "chassis":
        return "chassis_1"
    if label == "engine":
        return "engine_1"
    if label == "sit":
        return "sit_1"
    return label

def create_set(image_paths, features, all_labels):
    my_dict = {
        "id": [],
        "tokens": [],
        "bboxes": [],
        "ner_tags": [],
        "image": []
    }

    for image_path in image_paths:
        label_path = Path(image_path).parent.parent / "labels" / (Path(image_path).stem + ".json")
        label_data = json.load(open(label_path))
        # add = True
        # for i in label_data:
            # if preprocess_label(i['label']) not in ALL_LABELS:
            #     print(preprocess_label(i['label']), label_path)
        #         add = False
            # if i['text'] == "":
            #     print(i['text'], label_path)
            #     add = False

        # if add:
        my_dict["id"].append(Path(image_path).stem)
        my_dict["image"].append(image_path)
        tokens, bboxes, ner_tags = [], [], []
        for i in label_data:
            if preprocess_label(i['label']) in all_labels and i['text'] != "" and len(i['bndbox']) != 0:
                # tokens.append(remove_accent(i['text']))
                tokens.append(i['text'])
                bboxes.append(i['bndbox'])
                ner_tags.append(preprocess_label(i['label']))
            else:
                print(preprocess_label(i['label']), label_path)

        my_dict["tokens"].append(tokens)
        my_dict["bboxes"].append(bboxes)
        my_dict["ner_tags"].append(ner_tags)

    return Dataset.from_dict(my_dict, features=features)


def load_dkx_dataset(image_folder):
    ALL_LABELS = ["address", "brand", "capacity", "chassis_1", "chassis_2", "color", "engine_1", "engine_2", "first_issue_date", "model", "name", "plate", "sit_1", "sit_2"]
    features = Features({
        'id': Value(dtype='string', id=None),
        'tokens': Sequence(Value(dtype='string', id=None)),
        'bboxes': Sequence(Sequence(Value(dtype='int64', id=None))),
        'ner_tags': Sequence(ClassLabel(names=ALL_LABELS)),
        'image': Image(mode=None, decode=True),
    })
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    train_ratio = 0.8
    train_image_paths, test_image_paths = split_train_test(image_paths, train_ratio)

    dataset_train = create_set(train_image_paths, features, ALL_LABELS)
    dataset_test = create_set(test_image_paths, features, ALL_LABELS)
    dataset = DatasetDict({"train": dataset_train, "test": dataset_test})
    return dataset

from datasets.features import ClassLabel
def get_data_info(dataset, label_column_name):
    features = dataset["train"].features

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        id2label = {k: v for k,v in enumerate(label_list)}
        label2id = {v: k for k,v in enumerate(label_list)}
    else:
        label_list = get_label_list(dataset["train"][label_column_name])
        id2label = {k: v for k,v in enumerate(label_list)}
        label2id = {v: k for k,v in enumerate(label_list)}
    return label_list, id2label, label2id