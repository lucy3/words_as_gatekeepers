'''
Input: replacements created from write_mask_preds.py
- these are several .npy files in nested folders, 
  one for each batch: tokens, probs, doc_ids, lengths, reps

Output: 
- a file for every token containing its position in replacements
- additional files containing lemma and case sensitivity maps
'''
# pylint: disable=import-error
import argparse
import json
from functools import partial
from multiprocessing import Pool, cpu_count
import os
from collections import defaultdict
import spacy

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer

from utils import tokenizer_params

def main(replacements_dir, outdir, dataset):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_params[dataset], use_fast=True)
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    tokens_to_index = set()
    with open(args.vocab_path, 'r') as infile: 
        for line in infile: 
            w = line.strip()
            tokens_to_index.add(w)
    
    # this assert is needed since we are appending to index in this script
    # there might be one "SKIP" file in the outdir which is fine
    assert len([i for i in os.listdir(outdir) if not i.startswith('SKIP')]) == 0, "Indexing cannot already exist because we are appending."
    for folder in os.listdir(replacements_dir): 
        this_dir = os.path.join(replacements_dir, folder)
        number_to_tokens_files = len([f for f in os.listdir(this_dir) if f.endswith('tokens.npy')])
        print(f"total {number_to_tokens_files} files.")

        #Doing this because index is too big for memory so saving.
        files_step = 1000
        file_ranges = list(range(0, number_to_tokens_files, files_step)) + [number_to_tokens_files+1]
        which_files = []
        for i in range(len(file_ranges)-1):
            which_files.append((file_ranges[i], file_ranges[i+1]))

        partial_index = partial(index,
            tokens_to_index=tokens_to_index,
            replacements_dir=this_dir,
            outdir=outdir,
            dataset=dataset, 
            tokenizer=tokenizer,
            this_split=folder)

        with Pool(cpu_count()) as p:
            list(tqdm(p.imap(partial_index, which_files), total=len(which_files)))
        
    group_word_forms(outdir, tokenizer)
    index_doc_IDs(replacements_dir, outdir)
    group_all_word_forms(tokenizer, outdir)
    if args.dataset == 's2orc' and args.input_ids_path: 
        get_non_training_ids(args.input_ids_path, outdir)
    
def get_non_training_ids(paper_ids_path, outdir): 
    '''
    We only induce sense clusters for FOS-stratified
    sample and Wikipedia 
    paper_ids_path is a path to a list of IDs we don't 
    want to use during WSI training, e.g. input_paper_ids/journal_analysis.txt
    '''
    paper_ids = []
    with open(paper_ids_path, 'r') as infile: 
        for line in infile: 
            paper_ids.append(line.strip())
    # the input was created by wsi_preprocessing.py
    inpath = os.path.join(outdir, 'SKIP_paperID_sent.json') 
    with open(inpath, 'r') as infile: 
        d = json.load(infile) # {unique_file_id: s2orc paper ID}
    d_rev = {}
    for k in tqdm(d): 
        d_rev[d[k]] = k
    with open(os.path.join(outdir, 'SKIP_leave_out_ids.txt'), 'w') as outfile: 
        for p_id in tqdm(paper_ids): 
            outfile.write(d_rev[p_id] + '\n')
    
def index_doc_IDs(replacements_dir, outdir): 
    '''
    doc_id: 
        inst_id or guid (first column) in input file 
    file_id: 
        batch number or i in write_mask_preds.py
        for example, a tokens file would be replacements/file_id-tokens.npy
        The keys of the inverted index include file_id, not doc_id. 
    '''
    index_dict = {} # {doc_id : file_id }
    for this_split in os.listdir(replacements_dir): 
        this_dir = os.path.join(replacements_dir, this_split)
        all_files = os.listdir(this_dir) 
        all_files = sorted([f for f in all_files if f.endswith('-doc_ids.npy')])
        for filename in all_files:
            file_id = filename.split('-doc_ids.npy')[0]
            unique_file_id = this_split + '_' + str(file_id) # e.g. metadata_0_1823
            file_docs = np.load(os.path.join(this_dir, filename))
            for pos, doc_id in enumerate(file_docs): 
                index_dict[this_split + '_' + str(doc_id)] = unique_file_id
    doc_inverted_index = os.path.join(outdir, f"SKIP_doc_index.json")
    with open(doc_inverted_index, 'w') as outfile: 
        json.dump(index_dict, outfile)
        
def group_all_word_forms(tokenizer, outdir): 
    '''
    This is for all words in the tokenizer's vocab,
    mapping words indices that share the same form to a base form's index
    '''
    print("Getting tokenizer vocab word form groups...")
    vocab = tokenizer.get_vocab()
    
    lower_to_upper_map = {} # {index of word : index of lowercase word} 
    for word, index in tqdm(vocab.items()):
        lowercase_word = word.lower()
        if lowercase_word in vocab and lowercase_word != word: 
            lower_index = vocab[lowercase_word]
            lower_to_upper_map[index] = lower_index
        else: 
            lower_to_upper_map[index] = index
        
    with open(os.path.join(outdir, "SKIP_lower_to_upper_all.json"), 'w') as outfile: 
        json.dump(lower_to_upper_map, outfile) 
        
    nlp = spacy.load("en_core_web_lg", disable=['ner', 'parser'])
    lemma_group = defaultdict(list) # {str_token : [word_IDs]}
    shortest_form = {} # {lemma : (idx, shortest form in vocab)}, this will be the form we use in map
    for str_token, index in tqdm(vocab.items()):
        spacy_token = nlp(str_token)[0]
        if spacy_token.lemma_ == '-PRON-':
            lemma = str_token
        else:
            lemma = spacy_token.lemma_
        if lemma not in shortest_form: 
            shortest_form[lemma] = (index, str_token)
        else: 
            if len(str_token) <= len(shortest_form[lemma][1]) and str_token.lower() == str_token: 
                # found a better base form
                shortest_form[lemma] = (index, str_token)
        lemma_group[lemma].append(index)
        
    lemma_map = {} # {index of word : index of shortest word that shares same lemma} 
    for lemma in lemma_group: 
        shortest_idx, _ = shortest_form[lemma]
        for index in lemma_group[lemma]: 
            lemma_map[index] = shortest_idx
     
    with open(os.path.join(outdir, "SKIP_lemma_all.json"), 'w') as outfile: 
        json.dump(lemma_map, outfile) 
        
def group_word_forms(outdir, tokenizer): 
    '''
    Groups word IDs that have the same form (case insensitivity, lemmatization)
    This is for target words we want to induce senses for only 
    '''
    print("Getting target vocab word form groups...")
    all_files = os.listdir(outdir)
    word_IDs = [int(f.replace('.jsonl', '')) for f in all_files if not f.startswith('SKIP')]
    
    # case sensitive
    identity_map = defaultdict(list) # {str_token : [word_IDs]}, no modification
    for wordID in tqdm(word_IDs): 
        str_token = tokenizer.convert_ids_to_tokens(int(wordID))
        identity_map[str_token].append(str(wordID))
    with open(os.path.join(outdir, "SKIP_identity.json"), 'w') as outfile: 
        json.dump(identity_map, outfile) 
    
    # case insensitive
    lower_to_upper_map = defaultdict(list) # {str_token : [word_IDs]}, lower to upper case
    for wordID in tqdm(word_IDs): 
        str_token = tokenizer.convert_ids_to_tokens(int(wordID)).lower()
        lower_to_upper_map[str_token].append(str(wordID))
    # put the lowercase word ID first, if it exists
    for str_token in lower_to_upper_map: 
        for idx, wordID in enumerate(lower_to_upper_map[str_token]): 
            if tokenizer.convert_ids_to_tokens(int(wordID)) == str_token and idx != 0: 
                ID_list = lower_to_upper_map[str_token]
                # swap with the first index
                temp = ID_list[idx]
                ID_list[idx] = ID_list[0]
                ID_list[0] = temp
                lower_to_upper_map[str_token] = ID_list
                break
    with open(os.path.join(outdir, "SKIP_lower_to_upper.json"), 'w') as outfile: 
        json.dump(lower_to_upper_map, outfile)
        
    # case insensitive and lemmatized
    nlp = spacy.load("en_core_web_lg", disable=['ner', 'parser'])
    lemma_map = defaultdict(list) # {str_token : [word_IDs]}
    shortest_form = {} # {lemma : shortest form}
    for wordID in tqdm(word_IDs): 
        str_token = tokenizer.convert_ids_to_tokens(int(wordID)).lower()
        spacy_token = nlp(str_token)[0]
        if spacy_token.lemma_ == '-PRON-':
            lemma = str_token
        else:
            lemma = spacy_token.lemma_
        if lemma not in shortest_form: 
            shortest_form[lemma] = str_token
        else: 
            if len(str_token) < len(shortest_form[lemma]): 
                shortest_form[lemma] = str_token
        lemma_map[lemma].append(str(wordID))
    # put the shortest form first
    for lemma in lemma_map: 
        for idx, wordID in enumerate(lemma_map[lemma]): 
            if tokenizer.convert_ids_to_tokens(int(wordID)) == shortest_form[lemma]: 
                ID_list = lemma_map[lemma]
                # swap with the first index
                temp = ID_list[idx]
                ID_list[idx] = ID_list[0]
                ID_list[0] = temp
                lemma_map[lemma] = ID_list
                break
    with open(os.path.join(outdir, "SKIP_lemma.json"), 'w') as outfile: 
        json.dump(lemma_map, outfile)

def index(which_files, tokens_to_index, replacements_dir, outdir, dataset, tokenizer, this_split):
    index_dict = {} # {token : {file_id : position in -tokens.npy file} }

    all_files = os.listdir(replacements_dir) 
    all_files = sorted([f for f in all_files if f.endswith('tokens.npy')])
    if which_files:
        all_files = all_files[which_files[0]:which_files[1]] # Keeping the dict in memory is too expensive.
    for filename in all_files:
        file_id = filename.split('-tokens.npy')[0]
        unique_file_id = this_split + '_' + str(file_id) # e.g. metadata_0_1823
        file_tokens = np.load(os.path.join(os.path.join(replacements_dir, filename)))
        tok_to_positions = {} # {token : positions in -tokens.npy file}
        for pos, token in enumerate(file_tokens):
            # the vocabulary is lowercase 
            str_token = tokenizer.convert_ids_to_tokens(int(token)).lower()
            if str_token not in tokens_to_index:
                continue
            if token not in tok_to_positions:
                tok_to_positions[token] = []
            tok_to_positions[token].append(int(pos))

        for token, token_valid_positions in tok_to_positions.items():
            if token not in index_dict:
                index_dict[token] = {unique_file_id: token_valid_positions}
            else:
                index_dict[token][unique_file_id] = token_valid_positions

    for token, positions in index_dict.items():
        token_outfile = os.path.join(outdir, f"{token}.jsonl")
        with open(token_outfile, 'a') as f: # note that this is appending
            f.write(json.dumps(positions)+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--replacements_dir", type=str, default="replacements")
    parser.add_argument("--outdir", type=str, default='inverted_index')
    parser.add_argument("--dataset", type=str, choices=['s2orc'])
    parser.add_argument("--input_ids_path", type=str, required=False)
    parser.add_argument("--vocab_path", type=str, required=True)

    args = parser.parse_args()

    main(args.replacements_dir, args.outdir, args.dataset)