import json
import os
import torch
from typing import List, Union

from .language_model import get_embedding


class DatasetTweet(object):
    def __init__(
        self,
        id: int,
        in_reply_to_status_id: int,
        embedding: torch.Tensor,
    ):
        super().__init__()
        self.id = id
        self.in_reply_to_status_id = in_reply_to_status_id
        self.embedding = embedding

    def create_pair(self, other) -> torch.Tensor:
        return torch.stack([self.embedding, other.embedding])


def load_tweets_in_dir(dir: str) -> List[DatasetTweet]:
    result = []
    tweet_files = [f.path for f in os.scandir(dir)]

    for path in tweet_files:
        with open(path, "r", encoding="utf-8") as json_file:
            tweet = json.load(json_file)
            embedding, _ = get_embedding(tweet["text"])
            result.append(
                DatasetTweet(
                    tweet["id"], tweet["in_reply_to_status_id"], embedding
                )
            )

    return result


def convert_annotations(annotation: dict, string: bool = False) -> int:
    if "misinformation" in annotation.keys() and "true" in annotation.keys():
        if (
            int(annotation["misinformation"]) == 0
            and int(annotation["true"]) == 0
        ):
            if string:
                label = "unverified"
            else:
                label = 2
        elif (
            int(annotation["misinformation"]) == 0
            and int(annotation["true"]) == 1
        ):
            if string:
                label = "true"
            else:
                label = 1
        elif (
            int(annotation["misinformation"]) == 1
            and int(annotation["true"]) == 0
        ):
            if string:
                label = "false"
            else:
                label = 0
        elif (
            int(annotation["misinformation"]) == 1
            and int(annotation["true"]) == 1
        ):
            print("OMG! They both are 1!")
            print(annotation["misinformation"])
            print(annotation["true"])
            label = None

    elif (
        "misinformation" in annotation.keys()
        and "true" not in annotation.keys()
    ):
        # all instances have misinfo label but don't have true label
        if int(annotation["misinformation"]) == 0:
            if string:
                label = "unverified"
            else:
                label = 2
        elif int(annotation["misinformation"]) == 1:
            if string:
                label = "false"
            else:
                label = 0

    elif (
        "true" in annotation.keys()
        and "misinformation" not in annotation.keys()
    ):
        print("Has true not misinformation")
        label = None
    else:
        print("No annotations")
        label = None

    return label


def label_to_name(label: int) -> str:
    assert label in [0, 1, 2]

    if label == 0:
        return "false"
    elif label == 1:
        return "true"
    else:
        return "unverified"


def pairs_from_structure(d: Union[dict, list], parent_id=None):
    pairs = []
    for key, value in d.items():
        if parent_id != None:
            pairs.append((parent_id, key))
        if isinstance(value, dict):
            pairs.extend(pairs_from_structure(value, key))
    return pairs
