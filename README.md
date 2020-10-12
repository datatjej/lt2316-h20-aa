# LT2316 H20 Assignment A1

Name: Tova Erbén

*****************************************************************************
NB: In order to run part 2 of the code on the MLTGPU, please do the following:

import nltk<br>
nltk.download('averaged_perceptron_tagger')

*****************************************************************************


## Notes on Part 1.

For the tokenization I replaced all hyphens by white space in order to better account for compound words, and then tokenized using the nltk.word_tokenizer. With the helper functions get_token_instance and get_ner_instance I created two list of lists (where each inner list represents a single ner or token instance): ner_list and data_list. In the get_ner_instance function, NERs are split up if the character span field happens to contain the ';' character, indiciating that the NER is a multispan NER. 

These lists were made into data frames using the helper function list2df. In this function I also assigned 15 percent of the unique sentences in 'Train' to 'Val'. The way I currently do it, run.ipynb will yield a warning about *"A value is trying to be set on a copy of a slice from a DataFrame"* - but to the best of my knowledge the code should still work. The get_y function takes approximately 5-10 minutes to run. Sorry for that. :/ 


## Notes on Part 2.

For features I simply used POS tags, since I figured NERs might be more likely to be preceeded by punctuation (like ':' in *'DRUG:'*) or prepositions (*'15 mg of...'*). I did the POS tagging using the nltk.pos_tag method (see note about necessary installation on the mltgpu above). I mapped the tags to an integer the same way I did with word2id dictionary in part 1 and used '0' as padding, which I also did in get_y of part 1. 
