hf_cache = "/nlp/data/huggingface_cache"

bias_type_to_subspace_types = {
        'gender': ['pronouns', 'extended', 'propernouns', 'all', 'or', 'and'],
        'race': ['name', 'geo', 'all', 'or', 'and', 'race']
    }

bias_type_to_subspace_types_for_all = { # the subspace types included in 'all' subspace  
        # since the pronouns words are completely included in the extended words,
        # here we just use the extended words to avoid duplication
        'gender': ['extended', 'propernouns'], 
        'race': ['name', 'geo']
    }

bias_type_to_subspace_types_for_conceptor_operatoin = { # the subspace types included in 'or' and 'and' subspace  
        'gender': ['pronouns', 'extended', 'propernouns'],
        'race': ['name', 'geo']
    }


bias_type_jsonfile2name = {
    'gender': {
        # filename: target words and attributes
        'sent-weat6': 'SEAT-6',          
        'sent-weat7': 'SEAT-7',          
        'sent-weat8': 'SEAT-8',          
        'sent-weat6b': 'SEAT-6b',          
        'sent-weat7b': 'SEAT-7b',          
        'sent-weat8b': 'SEAT-8b',          
    }, 
}

