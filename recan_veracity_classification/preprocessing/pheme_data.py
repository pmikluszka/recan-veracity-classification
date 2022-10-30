import gc
from itertools import chain
import json
import os
from os.path import abspath, dirname, exists, join
import pickle
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from typing import List, Optional

from .language_model import cleanup_lm
from .pheme_helpers import convert_annotations, load_tweets_in_dir, DatasetTweet

RAW_DATASET_PATH = abspath(join(dirname(__file__), "../../data/raw/pheme"))
EMBEDDINGS_PATH = abspath(join(dirname(__file__), "../../data/embeddings"))

PHEME5 = [
    "charliehebdo",
    "germanwings",
    "ferguson",
    "ottawashooting",
    "sydneysiege",
]

PHEME9 = [
    "charliehebdo",
    "germanwings",
    "ferguson",
    "ottawashooting",
    "sydneysiege",
    "putinmissing",
    "prince-toronto",
    "gurlitt",
    "ebola-essien",
]


class PhemeDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.X = [item[0] for item in data]
        self.y = [item[1] for item in data]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        masks = X.any(dim=-1)
        lengths = masks.sum(dim=-1)

        return X, ~masks, lengths, self.y[idx]


def collate_fn_pheme(data):
    X, masks, lengths, y = zip(*data)
    sample_lens = [len(x) for x in X]

    return (
        torch.cat(X),
        torch.cat(masks),
        torch.cat(lengths),
        sample_lens,
        torch.tensor(y).long(),
    )


def load_pheme(ds_type: str = "folds", test_event: Optional[str] = None):
    types = ["folds", "PHEME5", "PHEME9"]
    assert ds_type in types

    data = _load_pheme_embeddings()

    if ds_type == "folds":
        all_examples = list(chain(*data.values()))
        Y = [item[1] for item in all_examples]
        train, rest_examples, _, rest_Y = train_test_split(
            all_examples, Y, train_size=0.8, stratify=Y
        )
        test, dev = train_test_split(
            rest_examples, train_size=0.875, stratify=rest_Y
        )
    elif ds_type == "PHEME5":
        assert test_event in PHEME5

        for event in list(set(PHEME9) - set(PHEME5)):
            del data[event]

        test = data[test_event]
        del data[test_event]

        rest = list(chain(*data.values()))
        Y = [item[1] for item in rest]
        train, dev = train_test_split(rest, train_size=0.9, stratify=Y)
    elif ds_type == "PHEME9":
        assert test_event in PHEME9

        test = data[test_event]
        del data[test_event]

        rest = list(chain(*data.values()))
        Y = [item[1] for item in rest]
        train, dev = train_test_split(rest, train_size=0.9, stratify=Y)

    gc.collect()
    return PhemeDataset(train), PhemeDataset(dev), PhemeDataset(test)


def _load_pheme_embeddings(ignore_saved: bool = False):
    pickled_path = join(EMBEDDINGS_PATH, "pheme.pkl")
    if not os.path.exists(EMBEDDINGS_PATH):
        os.makedirs(EMBEDDINGS_PATH)

    if exists(pickled_path) and not ignore_saved:
        with open(pickled_path, "rb") as pickle_file:
            data = pickle.load(pickle_file)
    else:
        data = _parse_pheme()
        with open(pickled_path, "wb") as pickle_file:
            pickle.dump(data, pickle_file)

    return data


def _parse_pheme():
    print("Parsing PHEME - all events")
    data = {}
    events = [
        (f.name, f.path) for f in os.scandir(RAW_DATASET_PATH) if f.is_dir()
    ]
    for event_name, event_path in events:
        print(f"\tParsing event {event_name}")
        event_data = []
        event_path = join(event_path, "rumours")

        threads = [f.path for f in os.scandir(event_path) if f.is_dir()]
        for thread in tqdm(threads):
            annotation_path = join(thread, "annotation.json")
            with open(annotation_path) as json_file:
                annotation = json.load(json_file)
                y = convert_annotations(annotation)

            all_tweets: List[DatasetTweet] = []
            all_tweets.extend(load_tweets_in_dir(join(thread, "source-tweets")))
            source = all_tweets[0]
            all_tweets.extend(load_tweets_in_dir(join(thread, "reactions")))
            all_tweets.sort(key=lambda x: x.id)

            X = []
            if len(all_tweets) == 1:
                pair = source.create_pair(source)
                X.append(pair)
                # set to 'unverified' if no replies
                # same approach as PLAN (https://arxiv.org/pdf/2001.10667.pdf)
                y = 2
            else:
                for tweet in all_tweets:
                    if tweet.in_reply_to_status_id == None:
                        continue

                    try:
                        parent = next(
                            (
                                x
                                for x in all_tweets
                                if x.id == tweet.in_reply_to_status_id
                            )
                        )
                    except StopIteration:
                        # no parent -> reply to source
                        parent = source

                    pair = parent.create_pair(tweet)
                    X.append(pair)

            X = torch.stack(X)
            event_data.append((X, y))

        data[event_name] = event_data

    cleanup_lm()
    return data


class PHEMEDataModule(LightningDataModule):
    def __init__(
        self,
        ds_type: str = "folds",
        test_event: Optional[str] = "charliehebdo",
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        self.num_classes = 3
        self.ds_type = ds_type
        self.test_event = test_event
        self.batch_size = batch_size

        # not in setup so we can get the dim_input from data
        self.train_dataset, self.val_dataset, self.test_dataset = load_pheme(
            self.ds_type, self.test_event
        )
        self.dim_input = self.train_dataset[0][0].shape[-1]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn_pheme,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn_pheme,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn_pheme,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn_pheme,
        )

    @staticmethod
    def add_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("PHEME")
        parser.add_argument("--ds_type", type=str, default="folds")
        parser.add_argument("--test_event", type=str, default="charliehebdo")
        return parent_parser


if __name__ == "__main__":
    train, dev, test = load_pheme("PHEME5", "charliehebdo")
    print(f"sample: {train[0][0]}")
    print(f"sample: {train[0][0].shape}")
    print(f"masks: {train[0][1]}")
    print(f"masks: {train[0][1].shape}")
    print(f"lengths: {train[0][2]}")
    print(f"lengths: {train[0][2].shape}")
    print(f"y: {train[0][3]}")

    train_dl = DataLoader(train, 4, collate_fn=collate_fn_pheme)
    for (X, masks, lengths, sample_lens, y) in train_dl:
        print(
            X,
            masks,
            lengths,
            sample_lens,
            y,
            sep="\n",
        )
        print(
            X.shape,
            masks.shape,
            lengths.shape,
            len(sample_lens),
            y.shape,
            sep="\n",
        )
        break
