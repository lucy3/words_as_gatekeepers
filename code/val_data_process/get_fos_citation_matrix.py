"""
Get a citation matrix of 
level 1 FOS + 1 (unknown) x level 1 FOS + 1 (unknown)

column = FOS that are citing, row = FOS that are cited
"""
import os
from collections import Counter, defaultdict
import json
import time
import multiprocessing
import boto3
from tqdm import tqdm
import subprocess
import pandas as pd
import csv

ROOT = '/home/lucyl/language-map-of-science/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
S3_S2ORC_BUCKET = 'ai2-s2-s2orc'
S3_S2ORC_PREFIX = '20200705v1/full/metadata/'
METADATA = '/net/nfs.cirrascale/s2-research/lucyl/full_metadata/'

def process_batch(batch): 
    start = time.time()
    short_name = batch['short_name']
    s3_url = batch['s3_infile']
    download_target_dir = batch['local_folder']
    download_target_path = batch['local_folder'] + short_name + '.jsonl.gz'
    gunzip_target_path = download_target_path.replace('.gz', '')
    outfolder = batch['out_folder']

    # download
    download = False 
    if not os.path.exists(download_target_path): 
        download = True
        args = ['/home/lucyl/bin/aws', 's3', 'cp', s3_url, f'{download_target_dir}']
        subprocess.run(args)

    args = ['gunzip', '-f', download_target_path]
    subprocess.run(args)

    file_length = sum(1 for line in open(gunzip_target_path, 'r'))

    result_d = Counter() # {cited FOS + '@' + citing FOS : count}
    
    # maybe this will make things faster
    memoized_level1 = defaultdict(list) # {paper_id : [memoized fos level 1 categories]}
    
    with open(gunzip_target_path, 'r') as infile: 
        for line in tqdm(infile, total=file_length): 
            d = json.loads(line)
            if not d['has_inbound_citations']: continue
            year = d['year']
            if year is None or year == 'null' or year < 2000 or year >= 2020: 
                continue
            paper_id = d['paper_id']
            if paper_id not in memoized_level1: 
                curr_fos = paper_fos[paper_id].split(',')
                curr_fos = [field for field in curr_fos if field in level_to_topics['1']]
                memoized_level1[paper_id] = curr_fos
            else: 
                curr_fos = memoized_level1[paper_id]
                
            all_other_fos = Counter() # { fos : count } 
            for other_id in d['inbound_citations']:
                if other_id not in memoized_level1: 
                    if other_id not in paper_fos: continue
                    other_fos = paper_fos[other_id].split(',')
                    other_fos = [field for field in other_fos if field in level_to_topics['1']]
                    memoized_level1[other_id] = other_fos
                    all_other_fos.update(other_fos)
                else: 
                    other_fos = memoized_level1[other_id]
                    
            for fos in curr_fos: 
                for o_fos in all_other_fos: 
                    key = fos + '@' + o_fos
                    result_d[key] += all_other_fos[o_fos]
    
    with open(outfolder + batch['short_name'] + '.json', 'w') as outfile: 
        json.dump(result_d, outfile)
            
    end = time.time()
    os.remove(gunzip_target_path)
    
def get_paper_fos(): 
    time_fos_df = pd.read_parquet(LOGS + 'citing_papers/')
    paper_fos = dict(zip(time_fos_df.paper_id, time_fos_df.fos))
    return paper_fos
                            
def get_level_topics(): 
    level_to_topics = defaultdict(set)
    with open(DATA + 'fos_level.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            level_to_topics[row['level']].add(row['displayname'].lower())
    return level_to_topics

def main(): 
    os.makedirs(LOGS + 'citation_matrix/', exist_ok=True)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(S3_S2ORC_BUCKET)
    filenames = []
    for obj in bucket.objects.filter(Prefix=S3_S2ORC_PREFIX + 'metadata'): 
        filenames.append(obj.key)

    print("making batches...")
    batches = [{
        'short_name': filename.split('/')[-1].replace('.jsonl.gz', ''), 
        's3_infile': f's3://ai2-s2-s2orc/{filename}',
        'local_folder': METADATA,
        'out_folder': LOGS + 'citation_matrix/',
    } for filename in sorted(filenames)]

    remaining_batches = [batch for batch in batches]
    print("Batches to do:", len(remaining_batches))
   
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 3) as p:
        p.map(process_batch, remaining_batches)

    print("done") 

if __name__ == '__main__':
    paper_fos = get_paper_fos()
    level_to_topics = get_level_topics()
    main()

