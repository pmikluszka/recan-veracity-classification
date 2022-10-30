# ReCAN model for claim veracity classification

This repository contains code implementing **ReCAN** (***Re****ply-based **C**o-**A**ttention **N**etwork*) model proposed in my master's thesis "Detecting false claims in social media discussions" as well as code allowing to conduct experiments on the *PHEME* [[1]](#1) dataset.

- [Model Description](#model-description)
- [Quick Start](#quick-start-guide)
- [References](#references)

## Model Description

## Quick Start Guide

### Install prerequisites

This code uses [poetry](https://python-poetry.org/) to create and manage python virtual environment.

Initialize dependencies by running `poetry install`.

Activate virtual environment by running `poetry shell`.

### Dataset

This code relies on the *PHEME* dataset which can be downloaded [HERE](https://figshare.com/articles/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078). The data should be placed in `data/raw` directory.

### Running scripts



## References

<span id="1">[1]</span>
A. Zubiaga, M. Liakata, R. Procter, G. Wong Sak Hoi, P. Tolmie.
[Analysing how people orient to and spread rumours in social media by looking at conversational threads.](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0150989&type=printable)
PLOS ONE, 11(3):1–29, 2016.

<span id="2">[2]</span>
Lianwei Wu, Yuan Rao, Yongqiang Zhao, Hao Liang, and Ambreen Nazir. 2020.
[DTCA: Decision Tree-based Co-Attention Networks for Explainable Claim Verification.](https://aclanthology.org/2020.acl-main.97)
In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1024–1035, Online. Association for Computational Linguistics.
