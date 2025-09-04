import json
import os

class WEATLists(object):
    W_1_Flowers = ['aster', 'clover', 'hyacinth', 'marigold', 'poppy', 'azalea', 'crocus', 'iris', 'orchid', 'rose', 'bluebell', 'daffodil', 'lilac', 'pansy', 'tulip', 'buttercup', 'daisy', 'lily', 'peony', 'violet', 'carnation', 'gladiola' ,'magnolia', 'petunia', 'zinnia']
    W_1_Insects = ['ant', 'caterpillar', 'flea', 'locust', 'spider', 'bedbug', 'centipede', 'fly', 'maggot', 'tarantula', 'bee', 'cockroach', 'gnat', 'mosquito', 'termite', 'beetle', 'cricket', 'hornet', 'moth', 'wasp', 'blackfly', 'dragonfly', 'horsefly', 'roach', 'weevil']
    W_1_Pleasant = ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation']
    W_1_Unpleasant = ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'divorce', 'jail', 'poverty', 'ugly', 'cancer', 'kill', 'rotten', 'vomit', 'agony', 'prison']

    W_2_Instruments = ['bagpipe', 'cello', 'guitar', 'lute', 'trombone', 'banjo', 'clarinet', 'harmonica', 'mandolin', 'trumpet', 'bassoon', 'drum', 'harp', 'oboe', 'tuba', 'bell', 'fiddle', 'harpsichord', 'piano', 'viola', 'bongo', 'flute', 'horn', 'saxophone', 'violin']
    W_2_Weapons = ['arrow', 'club', 'gun', 'missile', 'spear', 'axe', 'dagger', 'harpoon', 'pistol', 'sword', 'blade', 'dynamite', 'hatchet', 'rifle', 'tank', 'bomb', 'firearm', 'knife', 'shotgun', 'teargas', 'cannon', 'grenade', 'mace', 'slingshot', 'whip']
    W_2_Pleasant =  W_1_Pleasant
    W_2_Unpleasant = W_1_Unpleasant

    W_3_Unused_full_list_European_American_names = ['Adam', 'Chip', 'Harry', 'Josh', 'Roger', 'Alan', 'Frank', 'Ian', 'Justin', 'Ryan', 'Andrew', 'Fred', 'Jack', 'Matthew', 'Stephen', 'Brad', 'Greg', 'Jed', 'Paul', 'Todd', 'Brandon', 'Hank', 'Jonathan', 'Peter', 'Wilbur', 'Amanda', 'Courtney', 'Heather', 'Melanie', 'Sara', 'Amber', 'Crystal', 'Katie', 'Meredith', 'Shannon', 'Betsy', 'Donna', 'Kristin', 'Nancy', 'Stephanie', 'Bobbie-Sue', 'Ellen', 'Lauren', 'Peggy', 'Sue-Ellen', 'Colleen', 'Emily', 'Megan', 'Rachel', 'Wendy']
    W_3_European_American_names = ['Adam', 'Harry', 'Josh', 'Roger', 'Alan', 'Frank', 'Justin', 'Ryan', 'Andrew', 'Jack', 'Matthew', 'Stephen', 'Brad', 'Greg', 'Paul', 'Jonathan', 'Peter', 'Amanda', 'Courtney', 'Heather', 'Melanie', 'Katie', 'Betsy', 'Kristin', 'Nancy', 'Stephanie', 'Ellen', 'Lauren', 'Colleen', 'Emily', 'Megan', 'Rachel']
    W_3_Unused_full_list_African_American_names = ['Alonzo', 'Jamel', 'Lerone', 'Percell', 'Theo', 'Alphonse', 'Jerome', 'Leroy', 'Rasaan', 'Torrance', 'Darnell', 'Lamar', 'Lionel', 'Rashaun', 'Tyree', 'Deion', 'Lamont', 'Malik', 'Terrence', 'Tyrone', 'Everol', 'Lavon', 'Marcellus', 'Terryl', 'Wardell', 'Aiesha', 'Lashelle', 'Nichelle', 'Shereen', 'Temeka', 'Ebony', 'Latisha', 'Shaniqua', 'Tameisha', 'Teretha', 'Jasmine', 'Latonya', 'Shanise', 'Tanisha', 'Tia', 'Lakisha', 'Latoya', 'Sharise', 'Tashika', 'Yolanda', 'Lashandra', 'Malika', 'Shavonn', 'Tawanda', 'Yvette']
    W_3_African_American_names = ['Alonzo', 'Jamel', 'Theo', 'Alphonse', 'Jerome', 'Leroy', 'Torrance', 'Darnell', 'Lamar', 'Lionel', 'Tyree', 'Deion', 'Lamont', 'Malik', 'Terrence', 'Tyrone', 'Lavon', 'Marcellus', 'Wardell', 'Nichelle', 'Shereen', 'Ebony', 'Latisha', 'Shaniqua', 'Jasmine', 'Tanisha', 'Tia', 'Lakisha', 'Latoya', 'Yolanda', 'Malika', 'Yvette']
    W_3_Pleasant = ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation']
    W_3_Unpleasant = ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'bomb', 'divorce', 'jail', 'poverty', 'ugly', 'cancer', 'evil', 'kill', 'rotten', 'vomit']

    W_4_Unused_full_list_European_American_names = ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Jay', 'Matthew', 'Neil', 'Todd', 'Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Kristen', 'Meredith', 'Sarah']
    W_4_European_American_names = ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Matthew', 'Neil', 'Todd', 'Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Meredith', 'Sarah']
    W_4_Unused_full_list_African_American_names = ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tremayne', 'Tyrone', 'Aisha', 'Ebony', 'Keisha', 'Kenya', 'Latonya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha']
    W_4_African_American_names = ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tyrone', 'Aisha', 'Ebony', 'Keisha', 'Kenya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha']
    W_4_Pleasant = ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation']
    W_4_Unpleasant = ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'bomb', 'divorce', 'jail', 'poverty', 'ugly', 'cancer', 'evil', 'kill', 'rotten', 'vomit']

    W_5_Unused_full_list_European_American_names = ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Jay', 'Matthew', 'Neil', 'Todd', 'Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Kristen', 'Meredith', 'Sarah']
    W_5_European_American_names = ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Matthew', 'Neil', 'Todd', 'Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Meredith', 'Sarah']
    W_5_Unused_full_list_African_American_names = ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tremayne', 'Tyrone', 'Aisha', 'Ebony', 'Keisha', 'Kenya', 'Latonya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha']
    W_5_African_American_names = ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tyrone', 'Aisha', 'Ebony', 'Keisha', 'Kenya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha']
    W_5_Pleasant = ['joy', 'love', 'peace', 'wonderful', 'pleasure', 'friend', 'laughter', 'happy']
    W_5_Unpleasant = ['agony', 'terrible', 'horrible', 'nasty', 'evil', 'war', 'awful', 'failure']

    W_6_Male_names = ['John', 'Paul', 'Mike', 'Kevin', 'Steve', 'Greg', 'Jeff', 'Bill']
    W_6_Female_names = ['Amy', 'Joan', 'Lisa', 'Sarah', 'Diana', 'Kate', 'Ann', 'Donna']
    W_6_Career = ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career']
    W_6_Family = ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives']

    W_7_Math = ['math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition']
    W_7_Arts = ['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture']
    W_7_Male_terms = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']
    W_7_Female_terms = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']

    W_8_Science = ['science', 'technology', 'physics', 'chemistry', 'Einstein', 'NASA', 'experiment', 'astronomy']
    W_8_Arts = ['poetry', 'art', 'Shakespeare', 'dance', 'literature', 'novel', 'symphony', 'drama']
    W_8_Male_terms = ['brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him']
    W_8_Female_terms = ['sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'hers', 'her']

    W_9_Mental_disease = ['sad', 'hopeless', 'gloomy', 'tearful', 'miserable', 'depressed']
    W_9_Physical_disease = ['sick', 'illness', 'influenza', 'disease', 'virus', 'cancer']
    W_9_Temporary = ['impermanent', 'unstable', 'variable', 'fleeting', 'short-term', 'brief', 'occasional']
    W_9_Permanent = ['stable', 'always', 'constant', 'persistent', 'chronic', 'prolonged', 'forever']

    W_10_Young_peoples_names = ['Tiffany', 'Michelle', 'Cindy', 'Kristy', 'Brad', 'Eric', 'Joey', 'Billy']
    W_10_Old_peoples_names = ['Ethel', 'Bernice', 'Gertrude', 'Agnes', 'Cecil', 'Wilbert', 'Mortimer', 'Edgar']
    W_10_Pleasant = ['joy', 'love', 'peace', 'wonderful', 'pleasure', 'friend', 'laughter', 'happy']
    W_10_Unpleasant = ['agony', 'terrible', 'horrible', 'nasty', 'evil', 'war', 'awful', 'failure']

    WEFAT_1_Careers = ['technician', 'accountant', 'supervisor', 'engineer', 'worker', 'educator', 'clerk', 'counselor', 'inspector', 'mechanic', 'manager', 'therapist', 'administrator', 'salesperson', 'receptionist', 'librarian', 'advisor', 'pharmacist', 'janitor', 'psychologist', 'physician', 'carpenter', 'nurse', 'investigator', 'bartender', 'specialist', 'electrician', 'officer', 'pathologist', 'teacher', 'lawyer', 'planner', 'practitioner', 'plumber', 'instructor', 'surgeon', 'veterinarian', 'paramedic', 'examiner', 'chemist', 'machinist', 'appraiser', 'nutritionist', 'architect', 'hairdresser', 'baker', 'programmer', 'paralegal', 'hygienist', 'scientist']
    WEFAT_1_Female_attributes = W_7_Female_terms # ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']
    WEFAT_1_Male_attributes = W_7_Male_terms # ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']

    WEFAT_2_Androgynous_Names = ['Kelly', 'Tracy', 'Jamie', 'Jackie', 'Jesse', 'Courtney', 'Lynn', 'Taylor', 'Leslie', 'Shannon', 'Stacey', 'Jessie', 'Shawn', 'Stacy', 'Casey', 'Bobby', 'Terry', 'Lee', 'Ashley', 'Eddie', 'Chris', 'Jody', 'Pat', 'Carey', 'Willie', 'Morgan', 'Robbie', 'Joan', 'Alexis', 'Kris', 'Frankie', 'Bobbie', 'Dale', 'Robin', 'Billie', 'Adrian', 'Kim', 'Jaime', 'Jean', 'Francis', 'Marion', 'Dana', 'Rene', 'Johnnie', 'Jordan', 'Carmen', 'Ollie', 'Dominique', 'Jimmie', 'Shelby']
    WEFAT_2_Female_attributes = WEFAT_1_Female_attributes
    WEFAT_2_Male_attributes = WEFAT_1_Male_attributes

class Wordlist_Loader(object):
    
    def __init__(self, wordlist_folder):
        self.wordlist_folder = wordlist_folder

    def _load_gender_wordlists_survey(self):
        '''Gender word lists originally from the survey paper'''
        ## src: https://github.com/McGill-NLP/bias-bench/blob/0c4dbd5ba676c9eca1fbef8c2279eac53e8eae4b/bias_bench/dataset/inlp.py
        with open(f"{self.wordlist_folder}/survey/bias_attribute_words.json", "r") as f:
            attribute_words = json.load(f)["gender"]

        male_biased_token_set = set([words[0] for words in attribute_words])
        female_biased_token_set = set([words[1] for words in attribute_words])
        gender_list_survey = list(male_biased_token_set) + list(female_biased_token_set)

        return gender_list_survey

    def _load_gender_wordlists_txt(self):
        '''Gender word lists originally from ConceptorDebias'''

        male_vino_extra, female_vino_extra, male_gnGlove, female_gnGlove, male_cmu, female_cmu = [], [], [], [], [], []

        winoWordsPath = self.wordlist_folder + '/corefBias/WinoBias/wino/extra_gendered_words.txt'
        with open(winoWordsPath, "r+") as f_in:
            for line in f_in:
                male_vino_extra.append(line.split('\t')[0])
                female_vino_extra.append(line.strip().split('\t')[1])

        gnGloveMaleWordPath = self.wordlist_folder + '/gn_glove/wordlist/male_word_file.txt'
        with open(gnGloveMaleWordPath, "r+") as f_in:
            for line in f_in:
                male_gnGlove.append(line.strip())
        gnGloveFemaleWordPath = self.wordlist_folder + '/gn_glove/wordlist/female_word_file.txt'
        with open(gnGloveFemaleWordPath, "r+") as f_in:
            for line in f_in:
                female_gnGlove.append(line.strip())

        cmuMaleWordPath = self.wordlist_folder + '/cmu/male.txt'
        with open(cmuMaleWordPath, "r+") as f_in:
            for line in f_in:
                w = line.strip()
                if len(w)>0 and w[0] != '#':
                    male_cmu.append(w)
        cmuFemaleWordPath = self.wordlist_folder + '/cmu/female.txt'
        with open(cmuFemaleWordPath, "r+") as f_in:
            for line in f_in:
                w = line.strip()
                if len(w)>0 and w[0] != '#':
                    female_cmu.append(w)
        
        return male_vino_extra, female_vino_extra, male_gnGlove, female_gnGlove, male_cmu, female_cmu
        
    def _load_gender_wordlists_conceptor(self, male_vino_extra, female_vino_extra, male_gnGlove, female_gnGlove, male_cmu, female_cmu):
        gender_list_pronouns = WEATLists.W_7_Male_terms + WEATLists.W_7_Female_terms + WEATLists.W_8_Male_terms + WEATLists.W_8_Female_terms
        gender_list_pronouns = list(set(gender_list_pronouns))

        gender_list_extended = male_vino_extra + female_vino_extra + male_gnGlove + female_gnGlove
        gender_list_extended = list(set(gender_list_extended))

        gender_list_propernouns = male_cmu + female_cmu
        gender_list_propernouns = list(set(gender_list_propernouns))

        gender_list_all = gender_list_pronouns + gender_list_extended + gender_list_propernouns
        gender_list_all = list(set(gender_list_all))
        
        return gender_list_pronouns, gender_list_extended, gender_list_propernouns, gender_list_all
        
        
    def _load_race_wordlists(self):
    
        race_list_name = WEATLists.W_3_Unused_full_list_European_American_names + WEATLists.W_3_European_American_names + WEATLists.W_3_Unused_full_list_African_American_names + WEATLists.W_3_African_American_names + WEATLists.W_4_Unused_full_list_European_American_names + WEATLists.W_4_European_American_names + WEATLists.W_4_Unused_full_list_African_American_names + WEATLists.W_4_African_American_names + WEATLists.W_5_Unused_full_list_European_American_names + WEATLists.W_5_European_American_names + WEATLists.W_5_Unused_full_list_African_American_names + WEATLists.W_5_African_American_names
        race_list_name = list(set(race_list_name))

        # From survey paper
        race_list_geo = ['black', 'white', 'caucasian', 'asian', 'african', 'africa', 'america', 'asia', 'china', 'europe']
        
        return race_list_name, race_list_geo
    
    def _get_gender_words(self, male_vino_extra, female_vino_extra, male_gnGlove, female_gnGlove, male_cmu, female_cmu):
        male_words = WEATLists.W_7_Male_terms + WEATLists.W_8_Male_terms + male_vino_extra + male_gnGlove + male_cmu
        female_words = WEATLists.W_7_Female_terms +  WEATLists.W_8_Female_terms + female_vino_extra + female_gnGlove + female_cmu
        return male_words, female_words
    
    def _get_race_words(self, race_list_name, race_list_geo):
        race_words = list(set(race_list_name + race_list_geo))
        return race_words
    
    def get_wordlists_and_words(self):
        
        gender_list_survey = self._load_gender_wordlists_survey()
        male_vino_extra, female_vino_extra, male_gnGlove, female_gnGlove, male_cmu, female_cmu = self._load_gender_wordlists_txt()
        gender_list_pronouns, gender_list_extended, gender_list_propernouns, gender_list_all = self._load_gender_wordlists_conceptor(male_vino_extra, female_vino_extra, male_gnGlove, female_gnGlove, male_cmu, female_cmu)
        race_list_name, race_list_geo = self._load_race_wordlists()
        male_words, female_words = self._get_gender_words(male_vino_extra, female_vino_extra, male_gnGlove, female_gnGlove, male_cmu, female_cmu)
        race_words = self._get_race_words(race_list_name, race_list_geo)
        
        # Dictionary of lists
        subspace_type_to_wordlist = {
            #For gender debiasing
            'pronouns': gender_list_pronouns,
            'propernouns': gender_list_propernouns, 
            'extended': gender_list_extended, 
            'all': gender_list_all, 
            'survey': gender_list_survey,
            
            #For race deibasing
            'name': race_list_name,
            'geo': race_list_geo
        }
        
        return subspace_type_to_wordlist, male_words, female_words, race_words
    

class Corpus_Loader(object):
    def __init__(self, corpus_folder):
        self.corpus_folder = corpus_folder
   
    def get_corpora_dict(self):
        import nltk # version==3.2.5
        from nltk.corpus import brown
        brown_corpus_path = self.corpus_folder.split('/')[0]
        nltk.data.path.append(brown_corpus_path)
        if not os.path.exists(f"{brown_corpus_path}/brown"):
            nltk.download('brown', download_dir=brown_corpus_path)
        brown_corpus = brown.sents() # 57,340 sentences
        
        ## src: https://github.com/pliang279/sent_debias/blob/1be02d4bc39585497ad292d7adbd5bc295d445fc/debias-BERT/experiments/def_sent_utils.py
        sst_corpus_path = f'{self.corpus_folder}/sst.txt'
        sst_corpus = []
        for i, sent in enumerate(open(sst_corpus_path, 'r')):
            if i == 0:
                continue 
            try:
                sent = sent.split('\t')[1:]
                sent = ' '.join(sent)
            except:
                pass
            sent = sent.strip()
            sst_corpus.append(sent.split(' '))
            
        ## src: https://github.com/pliang279/sent_debias/blob/1be02d4bc39585497ad292d7adbd5bc295d445fc/debias-BERT/experiments/def_sent_utils.py
        reddit_corpus_path = f'{self.corpus_folder}/reddit.txt'
        reddit_data = open(reddit_corpus_path, 'r').read()
        reddit_corpus = []
        for i, sent in enumerate(reddit_data.split('\n')):
            if i == 0:
                continue 
            reddit_corpus.append(sent.strip().split(' '))
            
        corpus_type_to_corpus_instance = {
            'brown': brown_corpus, # length = 57340
            'sst': sst_corpus, # length = 11855
            'reddit': reddit_corpus, # length = 6002
        }
        
        return corpus_type_to_corpus_instance 

# Main method for testing
def main():
    wordlist_folder = 'data/wordlist'
    wordlist_loader = Wordlist_Loader(wordlist_folder)
    subspace_type_to_wordlist, male_words, female_words, race_words = wordlist_loader.get_wordlists_and_words()
    for subspace_type in subspace_type_to_wordlist:
        wordlist = subspace_type_to_wordlist[subspace_type]
        print(subspace_type, len(wordlist), wordlist[:10])
    print('male_words', len(male_words), male_words[:10])
    print('female_words', len(female_words), female_words[:10])
    print('race_words', len(race_words), race_words[:10])
    
    corpus_folder = 'data/corpora'
    corpus_loader = Corpus_Loader(corpus_folder)
    corpus_type_to_corpus_instance = corpus_loader.get_corpora_dict()
    for corpus_type in corpus_type_to_corpus_instance:
        corpus_instance = corpus_type_to_corpus_instance[corpus_type]
        print(corpus_type, len(corpus_instance), corpus_instance[:3])
    
if __name__ == "__main__":
    main()