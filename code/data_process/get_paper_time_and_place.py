"""
This script gets the following for ALL papers
in S2ORC between 2000-2019: 
- fos (aka place)
- time 
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

    result_d = { # for figuring out dataset and stratified sampling
        'paper_id': [], 
        'fos': [], # comma delimited strings
        'year': [], 
    }
    
    with open(gunzip_target_path, 'r') as infile: 
        for line in tqdm(infile, total=file_length): 
            d = json.loads(line)
            year = d['year']
            if year is None or year == 'null' or year < 2000 or year >= 2020: 
                continue
            paper_id = d['paper_id']
            result_d['paper_id'].append(paper_id)
            result_d['year'].append(year)
            
            mag_id = d['mag_id']
            if mag_id in mag_fos: 
                fos = mag_fos[mag_id]
            else: 
                fos = ['none']
            result_d['fos'].append(','.join(fos)) 
    
    df = pd.DataFrame.from_dict(result_d)
    df.to_parquet(outfolder + 
                  batch['short_name'] + '.parquet') 
            
    end = time.time()
    os.remove(gunzip_target_path)

def main(): 
    os.makedirs(LOGS + 'citing_papers/', exist_ok=True)

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
        'out_folder': LOGS + 'citing_papers/',
    } for filename in sorted(filenames)]

    remaining_batches = [batch for batch in batches]
    print("Batches to do:", len(remaining_batches))
   
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 3) as p:
        p.map(process_batch, remaining_batches)

    print("done") 

if __name__ == '__main__':
    print("Loading mag ID to FOS...")
    with open(LOGS + 'wiktionary/mag_fos.json', 'r') as infile: 
        mag_fos = json.load(infile)
    print("Done loading.")
    main()

