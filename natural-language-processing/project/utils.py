import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
      
      /home/ironman/D/python/nlp_Russia/natural-language-processing/week3/Starspace/modelSaveFile.tsv
    """
    
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    ########################
    #### YOUR CODE HERE ####
    ########################
    #embeddings_path ="/home/ironman/D/python/nlp_Russia/natural-language-processing/week3/Starspace/modelSaveFile.tsv" 
    lines = open(embeddings_path,"r").readlines()
    starspace_embeddings={}
    for line in lines:
        starspace_embeddings[line.strip().split("\t")[0]] = [float(x) for x in line.strip().split("\t")[1:]]
    embeddings = starspace_embeddings
    #embeddings_dim = 100
    embeddings_dim = len(list(embeddings.values())[0])
    return embeddings, embeddings_dim


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    
    # Hint: you have already implemented exactly this function in the 3rd assignment.
    
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' 
        embedding dim: size of the representation

        result: vector representation for the question
    """
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    #print(type(embeddings))
    question_vector = np.zeros((dim,), dtype=np.float64)
    if len(question.split())==0:
        return question_vector
    else:
        len_words_in_embeddings = 0
        for word in question.split():
            if word in embeddings:
                question_vector += embeddings[word]
                len_words_in_embeddings +=1
        if len_words_in_embeddings ==0:
            return question_vector
        else:
            return question_vector/len_words_in_embeddings


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
