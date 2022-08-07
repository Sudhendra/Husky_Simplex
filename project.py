import numpy as np 
import string
from multipledispatch import dispatch
import re
from collections import Counter
from scipy.sparse import csr_matrix

STOPWORDS = ['those', 'on', 'own', '’ve', 'yourselves', 'around', 'between', 'four', 'been', 'alone', 'off', 'am', 'then', 'other', 'can', 'regarding', 'hereafter', 'front', 'too', 'used', 'wherein', '‘ll', 'doing', 'everything', 'up', 'onto', 'never', 'either', 'how', 'before', 'anyway', 'since', 'through', 'amount', 'now', 'he', 'was', 'have', 'into', 'because', 'not', 'therefore', 'they', 'n’t', 'even', 'whom', 'it', 'see', 'somewhere', 'thereupon', 'nothing', 'whereas', 'much', 'whenever', 'seem', 'until', 'whereby', 'at', 'also', 'some', 'last', 'than', 'get', 'already', 'our', 'once', 'will', 'noone', "'m", 'that', 'what', 'thus', 'no', 'myself', 'out', 'next', 'whatever', 'although', 'though', 'which', 'would', 'therein', 'nor', 'somehow', 'whereupon', 'besides', 'whoever', 'ourselves', 'few', 'did', 'without', 'third', 'anything', 'twelve', 'against', 'while', 'twenty', 'if', 'however', 'herself', 'when', 'may', 'ours', 'six', 'done', 'seems', 'else', 'call', 'perhaps', 'had', 'nevertheless', 'where', 'otherwise', 'still', 'within', 'its', 'for', 'together', 'elsewhere', 'throughout', 'of', 'others', 'show', '’s', 'anywhere', 'anyhow', 'as', 'are', 'the', 'hence', 'something', 'hereby', 'nowhere', 'latterly', 'say', 'does', 'neither', 'his', 'go', 'forty', 'put', 'their', 'by', 'namely', 'could', 'five', 'unless', 'itself', 'is', 'nine', 'whereafter', 'down', 'bottom', 'thereby', 'such', 'both', 'she', 'become', 'whole', 'who', 'yourself', 'every', 'thru', 'except', 'very', 'several', 'among', 'being', 'be', 'mine', 'further', 'n‘t', 'here', 'during', 'why', 'with', 'just', "'s", 'becomes', '’ll', 'about', 'a', 'using', 'seeming', "'d", "'ll", "'re", 'due', 'wherever', 'beforehand', 'fifty', 'becoming', 'might', 'amongst', 'my', 'empty', 'thence', 'thereafter', 'almost', 'least', 'someone', 'often', 'from', 'keep', 'him', 'or', '‘m', 'top', 'her', 'nobody', 'sometime', 'across', '‘s', '’re', 'hundred', 'only', 'via', 'name', 'eight', 'three', 'back', 'to', 'all', 'became', 'move', 'me', 'we', 'formerly', 'so', 'i', 'whence', 'under', 'always', 'himself', 'in', 'herein', 'more', 'after', 'themselves', 'you', 'above', 'sixty', 'them', 'your', 'made', 'indeed', 'most', 'everywhere', 'fifteen', 'but', 'must', 'along', 'beside', 'hers', 'side', 'former', 'anyone', 'full', 'has', 'yours', 'whose', 'behind', 'please', 'ten', 'seemed', 'sometimes', 'should', 'over', 'take', 'each', 'same', 'rather', 'really', 'latter', 'and', 'ca', 'hereupon', 'part', 'per', 'eleven', 'ever', '‘re', 'enough', "n't", 'again', '‘d', 'us', 'yet', 'moreover', 'mostly', 'one', 'meanwhile', 'whither', 'there', 'toward', '’m', "'ve", '’d', 'give', 'do', 'an', 'quite', 'these', 'everyone', 'towards', 'this', 'cannot', 'afterwards', 'beyond', 'make', 'were', 'whether', 'well', 'another', 'below', 'first', 'upon', 'any', 'none', 'many', 'serious', 'various', 're', 'two', 'less', '‘ve']

def __modify(input_str):
    if type(input_str) == tuple or type(input_str) == list:
        if type(input_str) == tuple:
            input_str = list(input_str)
        input_str = ' '.join(input_str)
    return input_str
            
    
def extend_words(words):
        STOPWORDS.append(words)
    
def remove_words(words):
    unwanted = set(words)
    new_words = [item for item in STOPWORDS if item not in unwanted]
    STOPWORDS = new_words

def tokenize(x):
    input_str = __modify(x)
    token = input_str.split()
    return token
    
def word_counter(x):
    """
    Count the number of words in a string.
    returns: int
    Usage: string.word_counter() where string is an instance of Word class.
    """
    input_str = __modify(x)
    words = tokenize(input_str)
    words = np.array(words)
    dictionary = {}
    for word in words:
        index = np.where( words == word)
        index = np.array(index).flatten()
        dictionary[word] = len(index)
    return dictionary
    
def remove_stopwords(x):
    """
    Removes stopwords from a string.
    returns: A list type, containing tokens with the stopwords removed
    Usage: string.remove_stopwords() where string is an instance of Word class.
    NOTE: Use extend_words(words) and remove_words(words) methods of Word class to modify STOPWORDS.
    """
    input_str = __modify(x)
    words = [item for item in tokenize(input_str) if item.lower() not in STOPWORDS]
    return words
    
def join_stopwords(x):
    """
    Generate a new string without stopwords.
    returns: A string without the stopwords.
    Usage: string.join_stopwords() where string is an instance of Word class.
    """
    input_str = __modify(x)
    new_text = "".join(remove_stopwords(input_str))
    return new_text
# Suddhendra's end 

#Karans Part
class Clean:

    @dispatch(str)
    def remove_punctuation(s):
        c = ""
        for i in s:
            if i not in string.punctuation:
                c+=i

        return c

    @dispatch(list)
    def remove_punctuation(s):
        for i in range(len(s)):
            c = ""
            for t in s[i]:
                if t not in string.punctuation:
                    c+=t
            s[i] = c

        return s

    def stem(t):
        l = []
        for w in t:
            if w.endswith('ical'):
                l.append(w.replace('ical','ic'))

            elif w.endswith('ies'):
                l.append(w.replace('ies','y'))

            elif w.endswith('eed'):
                l.append(w.replace('eed','ee'))

            elif w.endswith('sses'):
                l.append(w.replace('sses','ss'))

            elif w.endswith('ization'):
                l.append(w.replace('ization','ize'))

            elif w.endswith('ation'):
                l.append(w.replace('ation','ate'))

            elif w.endswith('or'):
                l.append(w.replace('or','e'))

            elif w.endswith('iveness'):
                l.append(w.replace('iveness','ive'))

            elif w.endswith('fulness'):
                l.append(w.replace('fulness','ful'))

            elif w.endswith('ousness'):
                l.append(w.replace('ousness','ous'))

            elif w.endswith('ality'):
                l.append(w.replace('ality','al'))

            elif w.endswith('ivity' or 'ability' or 'bility'):
                l.append(re.sub('(ivity|ability|bility)$','',w))

            elif w.endswith('cacy'):
                l.append(w.replace('cacy','cate'))

            elif w.endswith('icity'):
                l.append(w.replace('icity','e'))

            elif w.endswith('alize'):
                l.append(w.replace('alize','al'))

            elif w.endswith('ence' or 'er' or 'ize' or 'ent' or 'ible' or 'able' or 'ance' or 'ness' or 'less' or 'ship' or 'ing' or 'er' or 'ers' or 's' or 'ly' or 'ment' or 'al' or 'ed' or 'ance' or 'ful' or 'ism' or 'liness'):
                l.append(re.sub('(ence|er|ize|ent|ible|able|ance|ness|less|ship|ing|ly|s|ers|ment|al|ed|ance|ful|ism|liness)$','',w))
                
        return l
#Arya's Part	
    @dispatch(str)
    def remove_symbol(st):
        """ 
        Removes Symbols from a String 
        Input: A String containing symbols
        returns: A String without symbols
        Usage: x.remove_symbol(String), where x is an instance of class Clean 
        """
        pattern = r"""[^A-Za-z0-9 ,.']+"""
        st1 = re.sub(pattern,'',st)
        return st1

    @dispatch(list)
    def remove_symbol(st):
        """ 
        Removes Symbols from all the Strings in the given list
        Input: A list of Strings containing Symbols
        returns: A List of Strings without symbols
        Usage: x.remove_symbol(List), where x is an instance of class Clean 
        """
        st1 = []
        for x in range(len(st)):
            pattern = r"""[^A-Za-z0-9 ,.']+"""
            st1.append(re.sub(pattern,'',st[x]))
        return st1

## Nikhils Part 
class Class_Vectorization:
    def __init__(self, input_str = None):
        
        self.input_str = input_str
        
        if type(self.input_str) == tuple:
            self.input_str = list(self.input_str)
            
        elif type(self.input_str) == str:
            self.input_str = self.input_str.split(". ")
            
        word = Word(self.input_str)
        dictionary = word.word_counter()
        self.vocab = np.array(list(dictionary.keys()))
            
    def BOW_fit_transform(self):
        """
        Creates a matrix with strings as rows and words as columns. This array would consist of frequency of words present in each string. 
        returns: A array with frequency of words.
        Usage: Vectorize.BOW_fit_transform() where Vectorize is an instance of Class_Vectorization class.
        """

        array = np.zeros((len(self.input_str),len(self.vocab)), dtype = int)
        i = 0
        for sentence in self.input_str:
            # array[i][0] = sentence

            sentence = sentence.split(" ")
            sentence = np.array(sentence)
            j = 0
            for word in self.vocab:
                index = np.where(sentence == word)
                if np.size(index)==0:
                    array[i][j]=0
                else:
                    array[i][j]= len(index)
                j+=1 
            i+=1
        return array
    
    def BOW_transform(self, test_str):        
        """
        Creates a matrix with strings as rows and words as columns.The list of words is generated using the values passed while creating the object.
        This array would consist of frequency of words which is present in the list of words and input string. 
        Input - A string of list 
        returns: A array with frequency of words.
        Usage: Vectorize.BOW_transform(input) where Vectorize is an instance of Class_Vectorization class.
        """
        
        if (type(test_str) == tuple )or (type(test_str) == str ):
            
            if type(test_str) == tuple:
                test_str = list(test_str)
                
            else:
                test_str = test_str.split(". ")
                
        array = np.zeros((len(test_str),len(self.vocab)), dtype = int)
        i = 0
        for sentence in test_str:
            sentence = sentence.split(" ")
            sentence = np.array(sentence)
            j = 0
            for word in self.vocab:
                index = np.where(sentence == word)
                if np.size(index)==0:
                    array[i][j]=0
                else:
                    array[i][j]= len(index)
                j+=1
            i+=1
        return array
#Arya's Part
    def fit(data):
        unique = set()
        for sent in data:
            for word in sent.split(' '):
                if len(word) >= 2:
                    unique.add(word)

        vocab = {}
        for index,word in enumerate(sorted(list(unique))):
            vocab[word] = index
        return vocab


    def custom_trans(data):
        """
        Creates an matrix containing count of words in a string, i.e Vectorization of text based on term frequency
        Input: A List of strings
        Returns: A matrix containing vectorization of text, where no of rows are the sentences and columns are unique words present in all the Strings.
        Usage: Vectorize.custom_trans(data), where Vectorize is instance of class
        """
        data = [x.lower() for x in data]
        vocab = fit(data)
        row,col,val = [],[],[]
        for ind,sent in enumerate(data):
            count_word = dict(Counter(sent.split(' ')))
            for word,count in count_word.items():
                if len(word) >= 2:
                    col_index = vocab.get(word)
                    if col_index >=0:
                        row.append(ind)
                        col.append(col_index)
                        val.append(count)
        x = csr_matrix((val, (row,col)), shape=(len(data),len(vocab))).toarray() #Creating Sparse Matrix Representation for Count Vectorization
        return x
