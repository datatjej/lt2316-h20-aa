
#basics
import pandas as pd
# Feel free to add any new code to this script
import torch
import nltk

def extract_features(data:pd.DataFrame, max_sample_length:int,  sample_length_dict:dict, id2word:dict):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
    device = torch.device('cuda:1')
    print("extracting features")
    id2pos = {}
    # divide df by splits
    df_train = data[data.split=='Train']
    unique_sent_train = list(df_train['sentence_id'].unique())
    print("Unique sent in Train: ", len(list(df_train['sentence_id'].unique()))) 
    #print("df_train size: ", df_train.size)
    df_val = data[data.split=='Val']
    #print("df_val size: ", df_val.size)
    print("Unique sent in Val: ", len(list(df_val['sentence_id'].unique()))) 
    df_test = data[data.split=='Test']
    #print("df_test size: ", df_test.size)
    print("Unique sent in Test: ", len(list(df_test['sentence_id'].unique()))) 
    
    #***************************************************************
    #max_sample_length, sample_length_dict = get_sample_lengths(data_df)
    
    train_pos, id2pos = get_pos(df_train, max_sample_length, sample_length_dict, id2pos, id2word, 'Train')
    test_pos, id2pos = get_pos(df_test, max_sample_length, sample_length_dict, id2pos, id2word, 'Test')
    val_pos, id2pos = get_pos(df_val, max_sample_length, sample_length_dict, id2pos, id2word, 'Val')
    
    print("Val pos-tagged sent:", len(val_pos))
    #print("Val[0]: ", val_pos[0])
    print("Test pos-tagged sent:", len(test_pos))
    print("Train pos-tagged sent:", len(train_pos))
    
    #print("train labels:", len(train_labels))
    train_tensor = torch.LongTensor(train_pos)
    train_tensor = train_tensor.to(device)
    
    test_tensor = torch.LongTensor(test_pos)
    test_tensor = test_tensor.to(device)
    
   
    val_tensor = torch.LongTensor(val_pos)
    val_tensor = val_tensor.to(device)
    
    print("id2pos: ", id2pos)
    
    return val_tensor, test_tensor, train_tensor
    
    #return three tensor of the following dimensions: NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM

def get_pos(df, max_sample_length, sample_lengths_dict, id2pos, id2word, split):
    
    pos_sentences = []
    
    df_as_list = df.values.tolist()
    
    sentence = []
    sentence_count = 0
    for df_row in df_as_list:
        sentence_id = df_row[0]
        sentence_length = sample_lengths_dict[sentence_id]
        token_id = df_row[1]
        sentence.append(token_id)
        if len(sentence) == sentence_length:
            #print("SENT:", len(sentence))
            wordified_sent = []
            for token_id in sentence:
                if token_id in id2word.keys():
                    wordified_sent.append(id2word[token_id])
                else:
                    wordified_sent.append("UNKNOWN")
            pos_tagged_sent = nltk.pos_tag(wordified_sent)
            pos_id_sent = []
            for pos_tuple in pos_tagged_sent:
                pos_id, id2pos = map_pos_to_id(pos_tuple[1], id2pos, split)
                #print("pos_tuple[1]", pos_tuple[1])
                pos_id_sent.append(pos_id)
            padded_sent = get_padding(pos_id_sent, max_sample_length)        
            #print("WORDIFIED:", wordified_sent)
            #print("POS_TAGGED SENT:", pos_tagged_sent) 
            #print("POS_ID SENT:", pos_id_sent)
            pos_sentences.append(padded_sent)
            sentence_count +=1
            sentence = []
    
    return pos_sentences, id2pos

def map_pos_to_id(pos, id2pos, split):
    res = False
    for key in id2pos:
        if(id2pos[key] == pos):
            res = True
            return key, id2pos
    if res == False:
        #if split == 'Val' or 'Test':
            #return default tag 'NN' = 1
            #return '1', id2pos
        #else:
        pos_id = len(id2pos)+1
        id2pos[pos_id] = pos
    return pos_id, id2pos
 
def get_padding(sentence, max_sample_length):
    #print("SENTENCE LABELS: ", sentence_labels)
    diff = max_sample_length - len(sentence)
    #print("DIFF: ", diff)
    if int(diff) == 0:
        #print("SENTENCE WITH NO DIFF: ", sentence_labels, "DIFF: ", diff)
        return sentence
    else:
        padding = [0] * diff
        sentence.extend(padding)
    return sentence