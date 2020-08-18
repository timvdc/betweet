#!/usr/bin/env python

import covidtools
import pandas as pd
import glob
import json
import numpy as np 
import torch

from snorkel.preprocess.nlp import SpacyPreprocessor

from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import itertools
from pycm import ConfusionMatrix

import os
import sys
from train import Train as RobBERTTrainer
from textdataset import TextDataset, load_and_cache_examples

fileList = ['../filter/tweets_covid.json']
fileList.extend(glob.glob('../1percent/2020_full/*'))

allTweet = []
for f in fileList:
    tweetListOfDict = covidtools.processTweetArchive(f)
    allTweet.extend(tweetListOfDict)

df_tweets = pd.DataFrame(allTweet)
#df_tweets = df_tweets.iloc[0:5000]

allDev = []
with open('dev_dataset/random100.json', 'rt') as inF:
    for line in inF:
        allDev.append(json.loads(line))

df_dev = pd.DataFrame(allDev)

df_train = df_tweets.loc[~df_tweets['id'].isin(df_dev['id'])]


#label mappings
BE = 0
NL = 1
ABSTAIN = -1

#spacy preprocessor for dutch
spacy_preproc = SpacyPreprocessor('text', 'doc',
                                  language='nl_core_news_sm', memoize=True,
                                  disable=['tagger', 'parser']
)

## RULES
@labeling_function()
def country_code(x):
    #country_code based on tweet location
    #precise but low coverage
    if x.country_code == 'BE':
        return BE
    elif x.country_code == 'NL':
        return NL
    else:
        return ABSTAIN

#three rules for country in user location
@labeling_function()
def location_be(x):
    if x.user_location == None:
        return ABSTAIN
    elif 'belg' in x.user_location.lower():
        return BE
    else:
        return ABSTAIN

@labeling_function()
def location_nl1(x):
    if x.user_location == None:
        return ABSTAIN
    elif 'nederland' in x.user_location.lower():
        return NL
    else:
        return ABSTAIN

@labeling_function()
def location_nl2(x):
    if x.user_location == None:
        return ABSTAIN
    elif 'netherlands' in x.user_location.lower():
        return NL
    else:
        return ABSTAIN


## rules based on community names
## should probably be one rule per community to differentiate between
## different probabilities for different community mentions

gemeentenBE = covidtools.loadGemeentenBE()
@labeling_function()
def gemeente_be(x):
    if x.user_location == None:
        return ABSTAIN
    elif any(g in x.user_location.lower() for g in gemeentenBE):
        return BE
    else:
        return ABSTAIN

gemeentenNL = covidtools.loadGemeentenNL()
@labeling_function()
def gemeente_nl(x):
    if x.user_location == None:
        return ABSTAIN
    elif any(g in x.user_location.lower() for g in gemeentenNL):
        return NL
    else:
        return ABSTAIN

## keywords with spacy tokenization
# def keyword_lookup(x, keywords, label):
#     if any([keyword == w.text.lower() for keyword in keywords for w in x.doc]):
#         return label
#     return ABSTAIN

## spacy named entities
## (slow and many mistakes)
# def keyword_lookup(x, keywords, label):
#     print(x.doc.ents)
#     if any([keyword in ent.text.lower() for keyword in keywords for ent in x.doc.ents]):
#         return label
#     return ABSTAIN


## without spacy tokenization (much faster)
def keyword_lookup(x, keywords, label):
    if any(keyword in x.text.lower() for keyword in keywords):
        return label
    return ABSTAIN

def make_keyword_lf_BE(keywords, label=BE):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}_BE",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
#        pre=[spacy_preproc]
    )

def make_keyword_lf_NL(keywords, label=NL):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}_NL",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
#        pre=[spacy_preproc]
    )

## check for community mentions in tweets
## this time one rule for each community (cfr. 'geel', 'ham', 'halen', ..)

allKeywordLFGemeentenBE = []
for g in gemeentenBE:
    allKeywordLFGemeentenBE.append(make_keyword_lf_BE(keywords=[g]))

allKeywordLFGemeentenNL = []
for g in gemeentenNL:
    allKeywordLFGemeentenNL.append(make_keyword_lf_NL(keywords=[g]))


#finally, keyword matching for typical dutch/flemish named
#entities/expressions
keywords_BE = [
    'belgi', 'vlaanderen', 'sciensano',
    'vanranstmarc', 'nva', 'hln_be',
    'vrtnws', 'vanguchtsteven', 'demorgen',
    'vtmnieuws', 'begov', 'openvld', 'cdenv',
    'destandaard', 'terzaketv', 'mondmasker'
]

keywords_NL = [
    'nederland', 'devolkskrant', '@nos',
    'wilders', 'pvv', 'rivm',
    'rutte', 'nunl', 'cda',
    'vvd', 'nrc', 'rtlnieuws',
    'kabinet', 'npo', 'telegraaf',
    'fvd', 'jesseklaver', 'mondkapje'
]

allKeywordNamedEntBE = []
for k in keywords_BE:
    allKeywordNamedEntBE.append(make_keyword_lf_BE(keywords=[k]))

allKeywordNamedEntNL = []
for k in keywords_NL:
    allKeywordNamedEntNL.append(make_keyword_lf_NL(keywords=[k]))



lfs = [
    country_code,
    location_be, location_nl1, location_nl2,
    gemeente_be, gemeente_nl,
    
]

lfs += allKeywordLFGemeentenBE
lfs += allKeywordLFGemeentenNL

lfs += allKeywordNamedEntBE
lfs += allKeywordNamedEntNL

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)

result = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
print(result)

from snorkel.labeling.model import MajorityLabelVoter

majority_model = MajorityLabelVoter(cardinality=2)
preds_train_majority = majority_model.predict(L=L_train)

from snorkel.labeling.model import LabelModel
label_model = LabelModel(cardinality=2, verbose=True, device='cuda')
#according to location data, BE tweets = 10-15%
label_model.fit(L_train=L_train, n_epochs=500, class_balance=[0.15, 0.85], log_freq=100, seed=82)
preds_train_label = label_model.predict(L=L_train)

L_dev = applier.apply(df=df_dev)
mapping = {'BE': 0, 'NL': 1}
Y_dev = np.array([mapping[i] for i in df_dev['label']])

majority_acc = majority_model.score(L=L_dev, Y=Y_dev, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")
label_model_acc = label_model.score(L=L_dev, Y=Y_dev, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")


from snorkel.labeling import filter_unlabeled_dataframe

df_train_filtered, preds_train_label_filtered = filter_unlabeled_dataframe(
    X=df_train, y=preds_train_label, L=L_train
)


##Discriminative model: RobBERT

# from transformers import RobertaTokenizer, RobertaModel
# tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robBERT-base")
# model = RobertaModel.from_pretrained("pdelobelle/robBERT-base")

# model.cuda()

# def encode_text(text):
#     input_ids = torch.tensor([tokenizer.encode(text)]).cuda()
#     return model(input_ids)[0].mean(1)[0].detach().cpu().numpy()

tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robBERT-base")
model = RobertaForSequenceClassification.from_pretrained("pdelobelle/robBERT-base")
logging.info("loaded RobBERT")

from train_config import Config
config = Config()
config.evaluate_dataset = "dev_dataset/random100_eval"

def evaluate(dataset, model):
    labelMapping = {'BE': 0, 'NL': 1}
    df = pd.read_pickle(dataset + ".labels.pickle")
    model.eval() # disable dropout etc.
    
    mask_padding_with_zero = True
    block_size = 512
    results = []
    for row in tqdm(df.iterrows(), total=len(df), mininterval=1, position=1, leave=True):
        index = row[0]
        sentence = row[1]['text']
        label = labelMapping[row[1]['label']]

        tokenized_text = tokenizer.encode(tokenizer.tokenize(sentence)[- block_size + 3 : -1])

        input_mask = [1 if mask_padding_with_zero else 0] * len(tokenized_text)

        pad_token = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        while len(tokenized_text) < block_size:
            tokenized_text.append(pad_token)
            input_mask.append(0 if mask_padding_with_zero else 1)

        batch = tuple(torch.tensor(t).to(torch.device("cuda")) for t in [tokenized_text[0 : block_size - 3], input_mask[0 : block_size- 3], [0], [1] if label else [0]])
        inputs = {"input_ids": batch[0].unsqueeze(0), "attention_mask": batch[1].unsqueeze(0), "labels": batch[3].unsqueeze(0)}
        with torch.no_grad():
            outputs = model(**inputs)

            results.append({"true": label, "predicted": outputs[1][0].argmax().item()})

    model.train() # make sure the model is back in training mode
    return results

train_dataset = load_and_cache_examples("roberta", tokenizer, df_train_filtered, preds_train_label_filtered)

model.train()
logging.info("Put RobBERT in training mode")

RobBERTTrainer.train(config, train_dataset, model, tokenizer, evaluate)
