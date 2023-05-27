"""
Counts words per community and for the "background" corpus
"""
import os
import json
from collections import defaultdict, Counter
import time
import multiprocessing
import boto3
from tqdm import tqdm
import subprocess
from transformers import BasicTokenizer
import pandas as pd
import random
import numpy as np
import re
from helper import get_journal_venue

ROOT = '/home/lucyl/language-map-of-science/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
S3_S2ORC_BUCKET = 'ai2-s2-s2orc'
S3_S2ORC_PREFIX = '20200705v1/full/metadata/'
METADATA = '/net/nfs.cirrascale/s2-research/lucyl/full_metadata/'

def process_batch(batch): 
    random.seed(0)
    start = time.time()
    short_name = batch['short_name']
    
    # big file paths
    s3_url = batch['s3_infile']
    download_target_dir = batch['local_folder']
    download_target_path = batch['local_folder'] + short_name + '.jsonl.gz'
    gunzip_target_path = download_target_path.replace('.gz', '')
    
    # output paths
    outpath = batch['out_folder'] + short_name + '.json'
    logpath = batch['log_folder'] + short_name + '.json'
    
    # misc
    papers_to_keep = batch['papers_to_keep']
    s2orc_fos = batch['s2orc_fos']

    download = False 
    if not os.path.exists(download_target_path): 
        download = True
        args = ['/home/lucyl/bin/aws', 's3', 'cp', s3_url, f'{download_target_dir}']
        subprocess.run(args)

    args = ['gunzip', '-f', download_target_path]
    subprocess.run(args)
    
    tokenizer = BasicTokenizer(do_lower_case=True)
    
    # outputs
    journal_count = Counter() # { journal : number of abstracts included }
    journal_wc = defaultdict(Counter) # {journal : { word: count}}
    journal_wc_set = defaultdict(Counter) # {journal : { word: count per abstract }}
    
    with open(gunzip_target_path, 'r') as infile: 
        for line in tqdm(infile): 
            d = json.loads(line)
            journal = d['journal']
            venue = d['venue']
            paper_id = d['paper_id']
            journal = get_journal_venue(paper_id, journal, venue, papers_to_keep)
            if not journal: 
                continue
            
            # get FOS 
            assert d['paper_id'] in s2orc_fos
            fos = s2orc_fos[d['paper_id']]
            
            abstract = d['abstract']
            title = d['title'] 
            
            title_abs = title + '\n\n' + abstract 
            tokens = tokenizer.tokenize(title_abs)
            # exclude words that don't contain letters (e.g. punctuation, numbers)
            tokens = [t for t in tokens if re.search('[a-z]', t)]
            set_tokens = list(set(tokens))
            
            for field in fos: 
                journal_count[field] += 1
                journal_wc[field].update(tokens)
                journal_wc_set[field].update(set_tokens)
            journal_count['background'] += 1
            journal_wc['background'].update(tokens)
            journal_wc_set['background'].update(set_tokens)
                      
    res = { 
        'set-True': journal_wc_set, 
        'set-False': journal_wc, 
        } 
    with open(outpath, 'w') as outfile: 
        json.dump(res, outfile)

    end = time.time()
    os.remove(gunzip_target_path)
    
    with open(logpath, 'w') as outfile: 
        json.dump(journal_count, outfile)

def count_words_per_input(fos_map_path='wiktionary/s2orc_fos.json'): 
    '''
    fos_map_path: which field of study map to use
    '''
    out_folder = LOGS + 'fos_word_counts/'
    log_folder = LOGS + 'fos_word_counts_logs/'

    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(S3_S2ORC_BUCKET)
    filenames = []
    for obj in bucket.objects.filter(Prefix=S3_S2ORC_PREFIX + 'metadata'): 
        filenames.append(obj.key)
        
    papers_to_keep = set()
    with open(DATA + 'input_paper_ids/fos_analysis.txt', 'r') as infile: 
        for line in infile: 
            papers_to_keep.add(line.strip())
            
    with open(LOGS + fos_map_path, 'r') as infile:
        s2orc_fos = json.load(infile) # {s2orc ID : [MAG FOS]}

    print("making batches...")
    batches = [{
        'short_name': filename.split('/')[-1].replace('.jsonl.gz', ''), 
        's3_infile': f's3://ai2-s2-s2orc/{filename}',
        'local_folder': METADATA,
        'out_folder': out_folder,
        'log_folder': log_folder,
        'papers_to_keep': papers_to_keep,
        's2orc_fos': s2orc_fos,
    } for filename in sorted(filenames)]

    
    print("Batches to do:", len(batches))
   
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 3) as p:
        p.map(process_batch, batches)

    print("done") 
    
def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def aggregate_counts(per_abstract=True): 
    in_folder = LOGS + 'fos_word_counts/'
    out_path = LOGS + 'word_counts_fos_set-' + str(per_abstract) + '.parquet'

    print("Aggregating counts across jsons into a parquet")
    all_counts = defaultdict(Counter) # {journal { word : count } }
    for filename in tqdm(os.listdir(in_folder)): 
        json_path = in_folder + filename
        with open(json_path, 'r') as infile:
            d = Counter(json.load(infile))
            key = 'set-' + str(per_abstract)

            for j in d[key]: 
                all_counts[j] += d[key][j]
    d = {
        'word': [], 
        'fos': [], 
        'count': [],  
    }

    for j in tqdm(all_counts): 
        for w in all_counts[j]: 
            d['word'].append(w)
            d['fos'].append(j)
            d['count'].append(all_counts[j][w])
    df = pd.DataFrame(data=d)
    df.to_parquet(out_path)  

def main(): 
    count_words_per_input()
    aggregate_counts()
    aggregate_counts(per_abstract=False)

if __name__ == '__main__':
    main()
