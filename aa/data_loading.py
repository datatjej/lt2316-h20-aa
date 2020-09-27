
#basics
import random
import pandas as pd
import torch
#extra:
import os
import re 
import nltk
import string
from glob import glob
from lxml import etree

class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
   

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):


    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)

    def _parse_data(self,data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        data_list = []
        ner_list = []
        vocab = [] #keeping track of unique words in the data
        data_dir = glob("{}/*".format(data_dir)) #glob returns a possibly-empty list of path names that match data_dir 
                                            #...in this case a list with the two subdirectories 'Test' and 'Train'                                           
        for subdir in data_dir: #looping through 'Test' and 'Train'
            split = os.path.basename(subdir) #get the directory name without path
            subdir = glob("{}/*".format(subdir))
            if split == 'Train':
                for folder in subdir:
                    folder = glob("{}/*".format(folder))
                    for xml_file in folder:
                        token_instances, ner_instances, vocab = self.parse_xml(xml_file, split, vocab)
                        data_list = data_list + token_instances
                        for instance in ner_instances:
                                if instance:
                                    ner_list.append(instance)
            elif split == 'Test':
                for folder in subdir:  #looping through 'Test for DDI Extraction task' and 'Test for DrugNER task'
                    folder = glob("{}/*".format(folder))
                    for subfolder in folder: #looping through 'DrugBank' and 'MedLine'
                        subfolder = glob("{}/*".format(subfolder))
                        for xml_file in subfolder:
                            token_instances, ner_instances, vocab = self.parse_xml(xml_file, split, vocab)
                            data_list = data_list + token_instances
                            for instance in ner_instances:
                                if instance:
                                    ner_list.append(instance)

        self.data_df, self.ner_df = self.list2df(data_list, ner_list) #turn lists into dataframes
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #display(data_df)
        #return data_df, ner_df
    
    def parse_xml(self, xml_file, split, vocab):    
        tree = etree.parse(xml_file)
        root = tree.getroot()
    
        token_instances = [] #save all token 
        ner_instances = []
    
        for elem in root: #loop over sentence tags
            if elem.tag == 'sentence':
                sent_id = elem.attrib['id'] #get sentence id
                text = elem.attrib['text']  #get the sentence as a string of text
                text = text.replace('-', ' ') #replaces all hyphens with whitespace for easier split of compound words
                char_pos = -1 #variable for keeping track of character-based positions of the words in the sentence
                nltk_tokens = nltk.word_tokenize(text)
                for token in nltk_tokens:
                    char_pos, token_instance, vocab  = self.get_token_instance(char_pos, sent_id, token, split, vocab)
                    token_instances.append(token_instance)
            for subelem in elem: #looping through children tags (i.e. 'entity', 'pair') of sentence_id
                if subelem.tag == 'entity':
                    ner_instance = self.get_ner_instance(sent_id, subelem)
                    for instance in ner_instance: #loop through list of returned NER instances
                        ner_instances.append(instance) #save them individually in the ner_instances list
        return token_instances, ner_instances, vocab
        
        
    def list2df(self, data_list, ner_list):
        data_df = pd.DataFrame.from_records(data_list, columns=['sentence_id', 'token', 'token_id', 'char_start_id', 'char_end_id', 'split'])
        data_df = data_df[~data_df['token'].isin(list(string.punctuation))] #remove tokens that are just punctuation 
        data_df.drop('token', inplace=True, axis=1) #remove 'token' column since it's not needed anymore
        #'inPlace=True' means we are working on the original df, 'axis=1' refers to the column axis
        train_samples = data_df[data_df['split']=='Train'].sample(frac=0.15) #sample 15 % of 'Train'-labeled rows
        train_samples.split='Val' #replace those 'Train' labels with 'Val'
        data_df.update(train_samples) #incorporate the modified train samples back into the original dataframe
        ner_df = pd.DataFrame.from_records(ner_list, columns=['sentence_id', 'ner_id', 'char_start_id', 'char_end_id'])
        return data_df, ner_df    

    def get_token_instance(self, char_pos, sent_id, token, split, vocab):
        char_pos += 1
        char_start = char_pos
        char_end = char_start + len(token)-1
        token_id, vocab = self.map_token_to_id(token, vocab)
        token_instance = [sent_id, token, token_id, char_start, char_end, split]
        char_pos=char_end+1 #increase by 1 to account for the whitespace between the current and the next word
        return char_pos, token_instance, vocab

    def get_ner_instance(self, sent_id, entity):
         #Problem of this approach: if a NER might be tokenized differently from the token dataframe
        ner_instances = []
        charOffset = entity.attrib['charOffset']
        #HAPPY PATH: if the character span is a single span:
        if ';' not in charOffset:
            char_start = charOffset.split('-')[0]
            char_end = charOffset.split('-')[1]
            ner_id = entity.attrib['type'] #getting the label: 'brand', 'drug', 'drug_n' or 'group'
            ner_instance = [sent_id, ner_id, char_start, char_end]
            return [ner_instance]
        #PATH OF DOOM: for multiword entities with several character spans:
        if ';' in charOffset:
            for span in charOffset.split(';'):
                ner_id = entity.attrib['type'] #getting the label: 'brand', 'drug', 'drug_n' or 'group'
                char_start = span.split('-')[0]
                char_end = span.split('-')[1]
                ner_instance = [sent_id, ner_id, char_start, char_end]
                ner_instances.append(ner_instance)
                #print("SPECIAL NER_INSTANCE: ", ner_instance)
        return ner_instances
    
    def map_token_to_id(self, token, vocab):
        vocab = vocab
        if token not in vocab:
            vocab.append(token)
        token_id = vocab.index(token)
        return token_id, vocab

    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
        pass


    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        pass


    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        pass


    def plot_ner_per_sample_distribution(self):        
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        pass


    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        pass



