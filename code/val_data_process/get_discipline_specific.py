'''
This script gets discipline specific
journals and papers

These are journals that are mostly
one level 1 FOS
and papers that only belong to one level 1 FOS
'''
import csv 
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
import json
import re
import string

ROOT = '/data0/lucy/language-map-of-science/'
LOGS = ROOT + 'logs/'
DATA = ROOT + 'data/'

def get_levels_to_topics(): 
    level_to_topics = defaultdict(set)
    with open(DATA + 'fos_level.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            level_to_topics[row['level']].add(row['displayname'].lower())
    return level_to_topics
            
def get_single_discipline_papers(level_to_topics): 
    multi_journals = ['national science review', 'nature', 'nature communications', 
                'philosophical transactions of the royal society a', 'plos one', 
                'proceedings of the national academy of sciences', 'proceedings of the royal society a', 
                'science', 'science advances', 'scientific reports'] 
    
    journal_df = pd.read_csv(DATA + 'input_paper_ids/journal_df.csv', index_col=0)
    journal_df = journal_df.sort_values('clean journal/venue') 
    groups = journal_df['clean journal/venue'].unique()
    ret = defaultdict(dict)
    single_discipline_journals = defaultdict(list)
    for j in tqdm(groups): 
        single_discip = False
        discip = None
        
        # match based on field in title
        for field in level_to_topics['1']:
            j = j.translate(str.maketrans('', '', string.punctuation))
            if re.search(re.compile(r'\b%s\b' % field, re.I), j): 
                discip = field
                single_discip = True
                single_discipline_journals[field].append(j)
                break
        
        start_idx = journal_df['clean journal/venue'].searchsorted(j, 'left')
        end_idx = journal_df['clean journal/venue'].searchsorted(j, 'right')
        this_df = journal_df[start_idx:end_idx]
        if not single_discip and j not in multi_journals: 
            # check if highly single disciplinary
            total_papers = len(this_df['paper id'].unique())
            fos_count = Counter(this_df['fos list'].to_list())
            for field in fos_count: 
                if field not in level_to_topics['1']: continue
                if fos_count[field] / total_papers > 0.8: 
                    discip = field
                    single_discip = True
                    single_discipline_journals[field].append(j)
                    break
                else: 
                    break
            if not single_discip: continue
        paper2fos = this_df.groupby('paper id')['fos list'].apply(list).to_dict()
        journal_papers = []
        for paper in paper2fos: 
            fos = [field for field in paper2fos[paper] if field in level_to_topics['1']]
            if len(fos) != 1: continue # must have one level 1 FOS
            if single_discip and fos[0] != discip: continue # must be same FOS as journal majority
            if not single_discip:
                discip = fos[0]
            journal_papers.append((paper, discip))
        if single_discip: 
            ret['single'][j] = journal_papers
        else: 
            ret['multi'][j] = journal_papers
            
    with open(LOGS + 'general_specific/papers_of_interest.json', 'w') as outfile: 
        json.dump(ret, outfile)
    with open(LOGS + 'general_specific/single_discp_journals.json', 'w') as outfile: 
        json.dump(single_discipline_journals, outfile)

def main(): 
    '''
    The output is
    {'multi' or 'single' : {'journal': [single discipline paper IDs]}}
    '''
    levels_to_topics = get_levels_to_topics()
    get_single_discipline_papers(levels_to_topics)

if __name__ == '__main__':
    main()