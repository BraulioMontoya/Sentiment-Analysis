import contractions
import inflect
import nltk
import re
import preprocessor
import unicodedata

def clean_text(text):
    text = preprocessor.clean(text)
    text = re.sub(r'RT', '', text)
    text = re.sub(r'\w+:\s', '', text)
    text = re.sub(r':\s', '', text)
    
    text = contractions.fix(text)
    
    return normalize(text)

def normalize(text):
    words = nltk.word_tokenize(text)
    new_text = ''
    
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        
        if new_word != '':            
            if new_word.isdigit():
                new_word = inflect.engine().number_to_words(new_word)
                
            new_word = unicodedata.normalize('NFKD', new_word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_word = new_word.lower()
            
            new_text += new_word + ' '
    
    return new_text

def lemmatize(text):
    words = nltk.word_tokenize(text)
    new_text = ''
    
    for word in words:
        new_word = nltk.WordNetLemmatizer().lemmatize(word, pos = 'v')
        new_text += new_word + ' '
    
    return new_text
