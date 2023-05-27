def keep_abstract(year, abstract, title, journal, venue, not_keep_set): 
    '''
    This must match the filtering in data_preprocessing.py
    and General Dataset Statistics so
    that the calculations of what % of abstracts to
    take for large journals is correct. 
    '''
    if year is None or year == 'null' or year < 2000 or year >= 2020: 
        return None
    if abstract is None or title is None: 
        return None
    if journal and journal != 'null': 
        journal = journal.strip().lower() # case insensitive
        journal = ' '.join([i for i in journal.split(' ') if not any(char.isdigit() for char in i)]).strip()
    else: 
        journal = ''
    if venue and venue != 'null': 
        venue = venue.strip().lower()
        venue = ' '.join([i for i in venue.split(' ') if not any(char.isdigit() for char in i)]).strip()
    else: 
        venue = ''

    if journal == '' and venue == '': 
        return None
    elif journal == '': 
        new_k = venue # use venue
    elif venue == '': 
        new_k = journal # use journal
    elif journal == venue: 
        new_k = journal # both same
    elif journal != venue: 
        new_k = journal # use journal
    
    if new_k in not_keep_set: 
        return None
    return new_k

def get_journal_venue(paper_id, journal, venue, papers_to_keep): 
    '''
    This must match General Dataset Statistics steps for standardizing journal/venue
    '''
    if paper_id not in papers_to_keep: 
        return None

    if journal and journal != 'null': 
        journal = journal.strip().lower() # case insensitive
        journal = ' '.join([i for i in journal.split(' ') if not any(char.isdigit() for char in i)]).strip()
    else: 
        journal = ''
    if venue and venue != 'null': 
        venue = venue.strip().lower()
        venue = ' '.join([i for i in venue.split(' ') if not any(char.isdigit() for char in i)]).strip()
    else: 
        venue = ''

    if journal == '' and venue == '': 
        return None
    elif journal == '': 
        new_k = venue # use venue
    elif venue == '': 
        new_k = journal # use journal
    elif journal == venue: 
        new_k = journal # both same
    elif journal != venue: 
        new_k = journal # use journal
    
    return new_k