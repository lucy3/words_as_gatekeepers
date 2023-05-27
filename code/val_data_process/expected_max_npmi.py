import json
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from tqdm import tqdm
import csv

ROOT = '/data0/lucy/language-map-of-science/'
LOGS = ROOT + 'logs/'
DATA = ROOT + 'data/'

def get_fos_hierarchy(): 
    level_to_topics = defaultdict(set)
    with open(DATA + 'fos_level.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            level_to_topics[row['level']].add(row['displayname'].lower())
    child_to_parents = defaultdict(set)
    with open(DATA + 'mag_parent_child.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            parent = row['parent_display_name'].lower()
            child = row['child_display_name'].lower()
            if parent in level_to_topics['0']: 
                if child in level_to_topics['1']: 
                    child_to_parents[child].add(parent)
    return level_to_topics, child_to_parents

def general_specific_exp(level_to_topics, child_to_parents): 
    with open(LOGS + 'general_specific/papers_of_interest.json', 'r') as infile: 
        general_specific_papers = json.load(infile)
    paper_type = {} # {paper: journal type}
    paper_discp = {} # {paper: discipline}
    for key in general_specific_papers: 
        for journal in tqdm(general_specific_papers[key]): 
            tups = general_specific_papers[key][journal]
            for tup in tups: 
                paper_discp[str(tup[0])] = tup[1]
                paper_type[str(tup[0])] = key
    
    max_npmi_path = LOGS + 'general_specific/paper_first_n_max/'
    cutoff = '0.1'
    infolder = os.path.join(max_npmi_path, cutoff)
    res = {
        'Field': [],
        'Journal/venue type': [],
        'cutoff': [],
        'm': [], 
        'Max NPMI': [],
    }
    for f in tqdm(os.listdir(infolder)): 
        with open(os.path.join(infolder, f), 'r') as infile: 
            d = json.load(infile) # {paper id: [max npmi of first n token for each index n]}
        for n in range(100): 
            for paper_id in d:
                for fos in child_to_parents[paper_discp[paper_id]]: 
                    journal_type = paper_type[paper_id]
                    max_npmi = max(d[paper_id][:n+1])
                    res['Field'].append(fos)
                    res['Journal/venue type'].append(journal_type)
                    res['cutoff'].append(cutoff)
                    res['m'].append(n)
                    res['Max NPMI'].append(max_npmi)
                    
    df = pd.DataFrame(data=res)
    df.to_parquet(LOGS + 'general_specific/expected_max_of_first_n.parquet')
        
def fos_sample_exp(level_to_topics, child_to_parents): 
    paper_discp = {} 
    with open(LOGS + 'wiktionary/s2orc_fos.json', 'r') as infile: 
        s2orc_fos = json.load(infile)

    for paper_id in s2orc_fos: 
        if len(s2orc_fos[paper_id]) == 1:
            if 'OTHER ' in s2orc_fos[paper_id][0]: continue
            paper_discp[paper_id] = s2orc_fos[paper_id][0]
    
    max_npmi_path = LOGS + 'fos_sample/paper_first_n_max/'
    cutoff = '0.1'
    infolder = os.path.join(max_npmi_path, cutoff)
    max_npmi_of_first_n_tokens = defaultdict(dict)
    for n in range(100): 
        max_npmi_of_first_n_tokens[n] = defaultdict(list) # {fos_cutoff : [maxes]}
    for f in os.listdir(infolder): 
        with open(os.path.join(infolder, f), 'r') as infile: 
            d = json.load(infile) # {paper id: [max npmi of first n token for each index n]}
        for n in tqdm(range(100)): 
            for paper_id in d:
                for fos in child_to_parents[paper_discp[paper_id]]: 
                    fos_cutoff = fos + '_' + cutoff
                    max_npmi = max(d[paper_id][:n+1])
                    max_npmi_of_first_n_tokens[n][fos_cutoff].append(max_npmi)
    expected_max_of_first_n = defaultdict(list) # {fos_cutoff : [expected maxes]}
    for n in range(100): 
        for fos_cutoff in max_npmi_of_first_n_tokens[n]: 
            expected_max = np.mean(max_npmi_of_first_n_tokens[n][fos_cutoff])
            expected_max_of_first_n[fos_cutoff].append(expected_max)
    with open(LOGS + 'fos_sample/expected_max_of_first_n.json', 'w') as outfile: 
        json.dump(expected_max_of_first_n, outfile)
            
def main(): 
    level_to_topics, child_to_parents = get_fos_hierarchy()
    general_specific_exp(level_to_topics, child_to_parents)
    fos_sample_exp(level_to_topics, child_to_parents) 

if __name__ == '__main__':
    main()