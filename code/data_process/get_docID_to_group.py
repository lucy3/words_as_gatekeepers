"""
Gets paper IDs to journals and FOS
"""
import os
import json
from collections import defaultdict, Counter
import time
import multiprocessing
import boto3
from tqdm import tqdm
import langid
import subprocess
from transformers import BasicTokenizer
import pandas as pd
import random
import numpy as np
import re
import csv
from helper import get_journal_venue
import argparse

ROOT = '/home/lucyl/language-map-of-science/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
S3_S2ORC_BUCKET = 'ai2-s2-s2orc'
S3_S2ORC_PREFIX = '20200705v1/full/metadata/'
METADATA = '/net/nfs.cirrascale/s2-research/lucyl/full_metadata/'
        
def get_documentID_maps(): 
    print("Getting docID to fos...")
    with open(LOGS + 'wiktionary/s2orc_fos.json', 'r') as infile:
        s2orc_fos = json.load(infile) # {s2orc ID : [MAG FOS]}
    with open(LOGS + 'inverted_index/SKIP_paperID_sent.json', 'r') as infile: 
        sents = json.load(infile) # {split_sentID : s2orc ID}
        
    # {split_sentID : [MAG FOS]}, e.g. split_sentID = metadata_12_32536
    docID2fos = defaultdict(list)
    for split_sentID in tqdm(sents): 
        s2orcID = sents[split_sentID]
        if s2orcID in s2orc_fos: 
            docID2fos[split_sentID] = s2orc_fos[s2orcID]
            
    with open(LOGS + 'sense_assignments_lemmed/docID2fos.json', 'w') as outfile: 
        json.dump(docID2fos, outfile)
            
    print("Getting docID to journal...")
    with open(DATA + 'input_paper_ids/s2orc_journal.json', 'r') as infile: 
        journal_s2orc = json.load(infile) # {journal : [s2orc IDs]}
        
    s2orc_journal = {}
    for j in tqdm(journal_s2orc): 
        for s2orcID in journal_s2orc[j]: 
            s2orc_journal[s2orcID] = j
    docID2journal = {}
    for split_sentID in tqdm(sents): 
        s2orcID = sents[split_sentID]
        if s2orcID in s2orc_journal: 
            docID2journal[split_sentID] = s2orc_journal[s2orcID]
            
    with open(LOGS + 'sense_assignments_lemmed/docID2journal.json', 'w') as outfile: 
        json.dump(docID2journal, outfile)
    
def main(): 
    get_documentID_maps()

if __name__ == '__main__':
    main()
