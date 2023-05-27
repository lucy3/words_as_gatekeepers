"""
Reservoir sampling of
X paragraphs
from each Wikipedia
"""
import random
import os
from tqdm import tqdm

ROOT = '/home/lucyl/language-map-of-science/'
DATA = ROOT + 'data/'

def sample_wikipedia(inpath, outpath, k=1000000): 
    random.seed(0)
    reservoir = [] 
    with open(inpath, 'r') as infile: 
        line_id = 0
        for line in infile: 
            if line_id % 1000000 == 0: 
                print("PROCESSED", line_id, "LINES")
            num_tokens = len(line.split())
            # avoid lines that are too short, as often these are non-sentence bullet points
            if num_tokens < 10: 
                line_id += 1
                continue
            # reservoir sample: 
            if len(reservoir) < k: 
                reservoir.append(line_id)
            else: 
                j = random.randrange(line_id + 1)
                if j < k: 
                    reservoir[j] = line_id
            line_id += 1
            
    reservoir = set(reservoir)
    os.makedirs(outpath, exist_ok=True)
    split_num = 0
    outfile = None
    with open(inpath, 'r') as infile: 
        line_id = 0
        for line in infile: 
            if line_id % 1000000 == 0:
                if outfile is not None: 
                    outfile.close()
                split_num += 1
                outfile = open(os.path.join(outpath, 'split_' + str(split_num)), 'w')
                print("WRITTEN OR SKIPPED", line_id, "LINES")
            if line_id in reservoir: 
                outfile.write(str(line_id) + '\t' + line.strip() + '\n')
            line_id += 1
    outfile.close() 
    return reservoir
    
def sample_subset_wikipedia(inpath, outpath, reservoir, k=1000000): 
    '''
    This is used so that the journal wiki dataset is a subset of the fos wiki dataset.
    '''
    random.seed(0)
    assert k < len(reservoir)
    new_reservoir = []
    for i, line_id in tqdm(enumerate(reservoir)): 
        if len(new_reservoir) < k: 
            new_reservoir.append(line_id)
        else: 
            j = random.randrange(i + 1)
            if j < k: 
                new_reservoir[j] = line_id         
    reservoir = set(new_reservoir)
    os.makedirs(outpath, exist_ok=True)
    split_num = 0
    outfile = None
    with open(inpath, 'r') as infile: 
        line_id = 0
        for line in infile: 
            if line_id % 1000000 == 0: 
                if outfile is not None: 
                    outfile.close()
                split_num += 1
                outfile = open(os.path.join(outpath, 'split_' + str(split_num)), 'w')
                print("WRITTEN OR SKIPPED", line_id, "LINES")
            if line_id in reservoir: 
                outfile.write(str(line_id) + '\t' + line.strip() + '\n')
            line_id += 1
    outfile.close() 

def main(): 
    fos_dataset_size = 11990654 # from data/wikipedia/fos_analysis.txt
    reservoir = sample_wikipedia(DATA + 'wikipedia/enwiki-20221001-pages-articles.txt', DATA + 'wikipedia/enwikifos/', k=2*fos_dataset_size)
    
if __name__ == '__main__':
    main()