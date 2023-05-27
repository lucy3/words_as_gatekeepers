tokenizer_params = {
    's2orc': 'globuslabs/ScholarBERT', # dataset: tokenizer model 
    }

def get_model_stopwords(tokenizer): 
    # this is copied from https://github.com/allenai/WSIatScale/blob/master/utils/special_tokens.py
    stop_words = 'ourselves hers between Between yourself Yourself but But again Again there There about About once Once during During out Out very Very having Having with With they They own Own an An be Be some Some for For do Do its Its yours Yours such Such into Into of Of most Most itself other Other off Off is Is am Am or Or who Who as As from From him Him each Each the The themselves until Until below Below are Are we We these These your Your his His through Through don Don nor Nor me Me were Were her Her more More himself Himself this This down Down should Should our Our their Their while While above Above both Both up Up to To ours had Had she She all All no No when When at At any Any before Before them Them same Same and And been Been have Have in In will Will on On does Does then Then that That because Because what What over Over why Why so So did Did not Not now Now under Under he He you You herself has Has just Just where Where too Too only Only myself which Which those Those after After few Few whom being Being if If theirs my My against Against by By doing Doing it It how How further Further was Was here Here than Than'.split(' ')
    token_IDs = tokenizer.convert_tokens_to_ids(stop_words)
    return token_IDs
    
    
def jaccard_score_between_elements(set1, set2):
    intersection_len = len(set1.intersection(set2))
    union_len = len(set1) + len(set2) - intersection_len
    return intersection_len / union_len