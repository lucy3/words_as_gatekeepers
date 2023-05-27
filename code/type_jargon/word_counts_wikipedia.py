"""
Input: Regular
and simple Wikipedia files
Output: 
parquet of "word", "dataset", "count"
"""
import os
import json
from collections import defaultdict, Counter
import time
from tqdm import tqdm
from transformers import BasicTokenizer
import pandas as pd
import random
import numpy as np
import re
import multiprocessing

ROOT = '/home/lucyl/language-map-of-science/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'

def count_words(inpath): 
    tokenizer = BasicTokenizer(do_lower_case=True)
    wc_set = Counter()
    wc = Counter()
    
    with open(inpath, 'r') as infile: 
        for line in infile: 
            text = ' '.join(line.strip().split('\t')[1:])
            tokens = tokenizer.tokenize(text)
            # exclude words that don't contain letters (e.g. punctuation, numbers)
            tokens = [t for t in tokens if re.search('[a-z]', t)]
            set_tokens = list(set(tokens))
            wc.update(tokens)
            wc_set.update(set_tokens)
            
    res = { 
        'set-True': wc_set, 
        'set-False': wc, 
        } 
    return res

def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main(): 
    wiki_files = { 'enwikifos': DATA + 'wikipedia/enwikifos/', 
                  'enwikijournal': DATA + 'wikipedia/enwikijournal/', 
    }
    wiki_counts = {}
    for dataset in wiki_files: 
        print("Counting for dataset", dataset)
        batches = os.listdir(wiki_files[dataset])
        batches = [os.path.join(wiki_files[dataset], b) for b in batches]
        counts = { 
                'set-True': Counter(), 
                'set-False': Counter(), 
            } 
        for chunk in chunks(batches, 30): 
            with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 3) as p:
                results = list(tqdm(p.imap(count_words, chunk), total=len(chunk)))
            for res in tqdm(results): 
                counts['set-True'] += res['set-True']
                counts['set-False'] += res['set-False']

        wiki_counts[dataset] = counts
    for set_type in wiki_counts['enwikifos']: 
        if 'True' in set_type: 
            out_path = LOGS + 'word_counts_wiki_set-True.parquet'
        else: 
            out_path = LOGS + 'word_counts_wiki_set-False.parquet'
        d = {
            'word': [], 
            'dataset': [], 
            'count': [],  
        }
        for dataset in wiki_counts: 
            for w in wiki_counts[dataset][set_type]: 
                d['word'].append(w)
                d['dataset'].append(dataset)
                d['count'].append(wiki_counts[dataset][set_type][w])
            
        df = pd.DataFrame(data=d)
        df.to_parquet(out_path)  

if __name__ == '__main__':
    main()