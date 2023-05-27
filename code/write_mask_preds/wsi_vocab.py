'''
Vocabulary creation
+ checking, how many vocab words are 
single word tokens in ScholarBERT and SciBERT? 

Output format: 
id,word,sentence
'''
from nltk.corpus import stopwords
import random
import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer
from collections import defaultdict

#ROOT = '/net/nfs2.s2-research/lucyl/language-map-of-science/'
ROOT = '/home/lucyl/language-map-of-science/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'

def create_vocab(total_counts, word_spread, num_groups, per_abstract=True, top_percentile=90, spread_percent=50): 
    '''
    Vocab creation
    '''
    stop_words = set(stopwords.words('english'))
    
    if per_abstract: 
        outfile_path = LOGS + 'sense_vocab/wsi_vocab_set_' + str(top_percentile) + '_' + \
            str(spread_percent) + '.txt'
    else: 
        outfile_path = LOGS + 'sense_vocab/wsi_vocab_' + str(top_percentile) + '_' + \
            str(spread_percent) + '.txt'
          
    cutoff = np.percentile(list(total_counts.values()), top_percentile)
    top_words = set()
    for w in total_counts: 
        if total_counts[w] < cutoff or not re.search('[a-z]', w): # word needs to contain a-z letters
            continue
        top_words.add(w)
            
    top_words = top_words - stop_words
    
    print('After freq cutoff:', len(top_words))
    
    vocab = set()
    for w in top_words: 
        if w not in word_spread: continue
        if len(word_spread[w]) >= (spread_percent / 100)*num_groups: 
            vocab.add(w)
            
    print('After spread + stopword cutoff:', len(vocab))
            
    with open(outfile_path, 'w') as outfile: 
        outfile.write('\n'.join(list(vocab)))
        outfile.write('\n')
    
def write_vocab_candidates(): 
    in_file = LOGS + 'word_counts_fos_set-True.parquet'
    fos_df = pd.read_parquet(in_file)
    wiki_file = LOGS + 'word_counts_wiki_set-True.parquet'
    wiki_df = pd.read_parquet(wiki_file)
    wiki_df = wiki_df[wiki_df['dataset'] == 'enwikifos']
    
    df = pd.concat([fos_df, wiki_df])
    total_counts = df.groupby(['word']).sum().to_dict()['count']
    word_spread = fos_df[fos_df['count'] > 20].groupby('word')['fos'].apply(list).to_dict()
    num_groups = len(set(df['fos'].to_list()))
    
    create_vocab(total_counts, word_spread, num_groups, top_percentile=98, 
                         spread_percent=50)
            
def try_wordpieces(model_hf_path, model_name, top_percentile=98, spread_percent=50): 
    '''
    This is used to estimate the wordpiece rate of different models. 
    '''
    vocab_path = LOGS + 'sense_vocab/wsi_vocab_set_' + str(top_percentile) + '_' + \
            str(spread_percent) + '.txt'
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_fast=True)
    wordpieces = set()
    words_with_whole_word_versions = defaultdict(list)
    avg_num_pieces = []
    vocab = set()
    with open(vocab_path, 'r') as infile: 
        for line in infile: 
            w = line.strip()
            pieces = tokenizer.tokenize(w)
            if len(pieces) > 1: 
                title_word = tokenizer.tokenize(w.title())
                upper_word = tokenizer.tokenize(w.upper())
                if len(title_word) == 1: 
                    words_with_whole_word_versions[w].append(w.title())
                if len(upper_word) == 1: 
                    words_with_whole_word_versions[w].append(w.upper())
                if len(words_with_whole_word_versions[w]) == 0: 
                    wordpieces.add(w)
                    avg_num_pieces.append(len(pieces))
            vocab.add(w)
            
    outpath = LOGS + 'sense_vocab/' + model_name + 'wordpiece_rate.txt'
    with open(outpath, 'w') as outfile: 
        outfile.write('# total:' + str(len(vocab)) + '\n')
        outfile.write('# wordpieces:' + str(len(wordpieces)) + '\n')
        outfile.write('# avg pieces for wordpiece tokens:' + str(np.mean(avg_num_pieces)) + '\n')
        outfile.write('\n'.join(list(wordpieces)))
        outfile.write('\n')
        for w in words_with_whole_word_versions: 
            if len(words_with_whole_word_versions[w]) == 0: continue
            outfile.write('VERSIONS\t' + w + '\t')
            for v in words_with_whole_word_versions[w]: 
                outfile.write(v + '\t')
            outfile.write('\n')
    
def main(): 
    write_vocab_candidates()
    #try_wordpieces('allenai/scibert_scivocab_uncased', 'scibert')
    #try_wordpieces('globuslabs/ScholarBERT', 'ScholarBERT')

if __name__ == '__main__':
    main()