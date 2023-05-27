"""
This file computes npmi of words in each journal. 

    PMI is defined as 
    log(p(word|journal) / p(word)) 
    or 
    log of 
    (frequency of word in journal j / # of words in journal j) 
    ______________________________________________________________ 
    (frequency of word in all j's / total number of words)
    This is then normalized by h(w, j), or
    
    -log p(w, j) = -log ((frequency of w in journal j) / (total number of words))
    as defined in Zhang et al. 2017.

"""
import os
import json
from collections import defaultdict, Counter
import time
from tqdm import tqdm
import pandas as pd
import math
import csv
import re
import spacy
from multiprocessing import Pool, cpu_count
from functools import partial

ROOT = '/home/lucyl/language-map-of-science/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'


def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def save_lemmas(): 
    '''
    @inputs: 
    - per_abstract: whether to count words once per abstract
    
    This is used for calculating type NPMI for FOS so that the words are
    comparable to sense NPMI. 
    ~20 minutes for Medicine S2ORC
    '''
    nlp = spacy.load("en_core_web_lg", disable=['ner', 'parser'])
    in_file = LOGS + 'word_counts_fos_set-True.parquet'
    
    df = pd.read_parquet(in_file)
    background_file = LOGS + 'word_counts_wiki_set-True.parquet'
    background = pd.read_parquet(background_file)
    vocab = df['word'].unique()
    other_vocab = background['word'].unique()
    vocab = set(vocab) | set(other_vocab)
    
    sense_vocab = set()
    with open(LOGS + 'sense_vocab/wsi_vocab_set_98_50.txt', 'r') as infile: 
        for line in infile: 
            sense_vocab.add(line.strip())
    
    lemma_storage = {} 
    for doc in tqdm(nlp.pipe(vocab, batch_size=32, n_process=3, disable=["parser", "ner"]), total=len(vocab)):
        if len(doc) != 1: 
            lemma_storage[doc.text] = doc.text
            continue
        if doc.text not in sense_vocab: # only lemmatize words we induce senses for
            lemma_storage[doc.text] = doc.text
            continue
        spacy_token = doc[0]
        if spacy_token.lemma_ == '-PRON-':
            lemma = spacy_token.text
        else:
            lemma = spacy_token.lemma_
        lemma_storage[spacy_token.text] = lemma
    os.makedirs(LOGS + 'sense_vocab/', exist_ok=True)
    with open(LOGS + 'sense_vocab/' + 'all_lemmas-for-type-npmi.json', 'w') as outfile: 
        json.dump(lemma_storage, outfile)
        
def calc_npmi_helper(batch): 
    this_df = batch['this_df']
    overall_total = batch['overall_total']
    total_counts = batch['total_counts']
    lemmatize = batch['lemmatize']
    j = batch['j']
    
    pmi_d = {}
    if lemmatize: 
        word_counts = this_df.groupby(['lemma'], sort=False).sum().to_dict()['count']
    else: 
        word_counts = pd.Series(this_df['count'].values, index=this_df.word).to_dict()
    word_counts = Counter(word_counts)
    total_j = sum(word_counts.values())
    
    for tup in word_counts.most_common(): 
        w = tup[0]
        c = tup[1]
        if c <= 20: 
            break # do not calculate npmi for rare words
        if not re.search('[a-z]', w): # word needs to contain a-z letters
            continue
        p_w_given_j = c / total_j
        p_w = total_counts[w] / overall_total
        pmi = math.log(p_w_given_j / p_w)
        h = -math.log(c / overall_total)
        pmi_d[w] = pmi / h
    return (pmi_d, word_counts, j)

def get_background(df, group_name, lemmatize=True, per_abstract=True, write_all=True): 
    background_file = LOGS + 'word_counts_wiki_set-' + str(per_abstract) + '.parquet'
    background = pd.read_parquet(background_file)
    
    if lemmatize: 
        print("Lemmatizing...")
        with open(LOGS + 'sense_vocab/' + 'all_lemmas-for-type-npmi.json', 'r') as infile: 
            lemma_storage = json.load(infile)
        df['lemma'] = df['word'].map(lemma_storage)
        df = df.drop(columns=['word'])
        background['lemma'] = background['word'].map(lemma_storage)
        background = background.drop(columns=['word'])
        
    groups = df[group_name].unique()
    
    if group_name == 'journal': 
        wiki_background = background[background['dataset'] == 'enwikijournal']
        s2orc_background = df
        background = pd.concat([wiki_background, s2orc_background])
        
        if not write_all: 
            with open(LOGS + 'data_counts/journal_abs_count.json', 'r') as infile: 
                journal_abs_counts = Counter(json.load(infile))
            # only show 200 journals' npmi scores
            groups = [tup[0] for tup in journal_abs_counts.most_common(200)]
    elif group_name == 'fos': 
        wiki_background = background[background['dataset'] == 'enwikifos']
        s2orc_background = df[df[group_name] == 'background']
        background = pd.concat([wiki_background, s2orc_background])
    else: 
        print("NOT IMPLEMENTED YET")
        return 
    
    if lemmatize: 
        total_counts = background.groupby(['lemma']).sum().to_dict()['count']
        df = df.groupby([group_name, 'lemma']).sum().reset_index()
    else: 
        total_counts = background.groupby(['word']).sum().to_dict()['count']
    overall_total = sum(total_counts.values())
    
    return total_counts, overall_total, groups

def calc_npmi(per_abstract=True, lemmatize=True, group_name='journal', 
              write_all=False): 
    '''
    The input parquet specified by in_file contains columns of 
    - group_name (e.g. 'journal' or 'fos')
    - word
    - count
    
    @inputs: 
    - per_abstract: whether to count words once per abstract
    - lemmatize: whether to calculate npmi for lemmas
    - group_name: the grouping or domain of documents, 'journal' or 'fos'
    - out_folder: where to write the output
    - write_all: a flag where we limit the number of journals being written to just 200 or all of them. 
    '''
    in_file = LOGS + 'word_counts_' + group_name + '_set-' + str(per_abstract) + '.parquet'
    out_folder = LOGS + 'type_npmi/' + group_name + \
                        '_set-' + str(per_abstract) + \
                        '_lemma-' + str(lemmatize) + '/' 
    
    os.makedirs(out_folder, exist_ok=True)

    df = pd.read_parquet(in_file)
    total_counts, overall_total, groups = get_background(df, group_name, lemmatize=lemmatize, 
                                                 per_abstract=per_abstract, write_all=write_all)
    
    print("Done with getting background corpus.")
    
    if group_name == 'journal' and write_all: 
        # need speed up trick
        df = df.sort_values(group_name) 
        batches = []
        for j in tqdm(groups): 
            start_idx = df['journal'].searchsorted(j, 'left')
            end_idx = df['journal'].searchsorted(j, 'right')
            batch = { 
                'j': j,
                'this_df': df[start_idx:end_idx],
                'overall_total': overall_total,
                'total_counts': total_counts,
                'lemmatize': lemmatize,
            }
            batches.append(batch)
    else:  
        batches = [{ 
                'j': j,
                'this_df': df[df[group_name] == j],
                'overall_total': overall_total,
                'total_counts': total_counts,
                'lemmatize': lemmatize,
            } for j in tqdm(groups)]
    
    print("Done making batches.")
    
    if write_all: 
        result_d = {
            'journal': [],
            'word': [],
            'npmi': [], 
            'count': [], 
        }

        with Pool(processes=cpu_count() // 3) as p:
            results = list(tqdm(p.imap(calc_npmi_helper, batches), total=len(batches)))
            for res in results: 
                pmi_d, word_counts, j = res
                for word_npmi in pmi_d.items(): 
                    result_d['journal'].append(j)
                    result_d['word'].append(word_npmi[0])
                    result_d['npmi'].append(word_npmi[1])
                    result_d['count'].append(word_counts[word_npmi[0]])
        df = pd.DataFrame(data=result_d)
        df.to_parquet(out_folder + 'ALL_scores.parquet') 
        
    else: 
        for batch in tqdm(batches): 
            pmi_d, word_counts, j = calc_npmi_helper(batch)
            
            with open(out_folder + batch['j'], 'w') as outfile: 
                sorted_d = sorted(pmi_d.items(), key=lambda kv: kv[1])
                writer = csv.writer(outfile)
                writer.writerow(['word', 'pmi', 'count'])
                for tup in sorted_d: 
                    writer.writerow([tup[0], str(tup[1]), str(word_counts[tup[0]])])

def number_punctuation_tester(): 
    '''
    To test if any npmi words are numbers / punctuation
    '''
    # number: [0-9] regex, replace with [NUM] token
    # punctuation: try string.punctuation first
    for f in os.listdir(LOGS + 'npmi_set/'): 
        with open(LOGS + 'npmi_set/' + f, 'r') as infile: 
            for line in infile: 
                w = line.split(',')[0]
                if not re.search('[a-z]', w): print(w)

def main(): 
    print("Calculating FOS NPMI") 
    calc_npmi(group_name='fos', lemmatize=False, per_abstract=False)
    save_lemmas()
    calc_npmi(group_name='fos')
    calc_npmi(group_name='fos', per_abstract=False)
    #print("Calculating journal/venue NPMI") 
    #calc_npmi(group_name='journal', lemmatize=False, write_all=True) 

if __name__ == '__main__':
    main()
