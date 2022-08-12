from project import *
from copy import copy

def Bag_of_Words_testing(b):  
    bow_ftransform = Vectorizer(b)
    arr = np.array([[1, 1 ,1 ,1 ,1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0 ,0 ,0, 0 ,0 ,0 ,1 ,0 ,1 ,1 ,1 ,1]])
    result = np.testing.assert_array_equal(bow_ftransform.BOW_fit_transform(),arr)
    return result

def count_vector_testing(data):
    bow_vector = Vectorizer()
    out = CountVectorizer().fit_transform(data).toarray()
    result = np.testing.assert_array_equal(bow_vector.cv_trans(data),out)
    return result

def TfIdf_fTransform_testing(input_str):  
    
    tfIdf_ftransform = Vectorizer(input_str)
    testList = [{'best': 0.0, 'pizza': 0.07952020911994373, 'people': 0.0, 'good': 0.0, 'loves': 0.07952020911994373, 'person': 0.0, 'delicious': 0.07952020911994373}, 
        {'best': 0.0, 'pizza': 0.0, 'people': 0.0, 'good': 0.03521825181113625, 'loves': 0.0, 'person': 0.09542425094393249, 'delicious': 0.0}, 
        {'best': 0.07952020911994373, 'pizza': 0.0, 'people': 0.07952020911994373, 'good': 0.029348543175946873, 'loves': 0.0, 'person': 0.0, 'delicious': 0.0}]
    sortedDict = []    
    for d in testList:            
        sortedDict.append(dict( sorted(d.items(), key=lambda x: x[0].lower()) )) 

    expectedOutputDF = pd.DataFrame(sortedDict)

    pd.testing.assert_frame_equal(tfIdf_ftransform.tfIdf_fit_transform(),expectedOutputDF, check_dtype=False)
    boolResult = expectedOutputDF.equals(tfIdf_ftransform.tfIdf_fit_transform())

    if(boolResult == True):
        return "Pass"
    else:
        return "Fail"    

def TfIdf_transform_testing(input_str):  

    tfIdf_ftransform = Vectorizer()
    testList = [{'best': 0.0, 'pizza': 0.07952020911994373, 'people': 0.0, 'good': 0.0, 'loves': 0.07952020911994373, 'person': 0.0, 'delicious': 0.07952020911994373}, 
        {'best': 0.0, 'pizza': 0.0, 'people': 0.0, 'good': 0.03521825181113625, 'loves': 0.0, 'person': 0.09542425094393249, 'delicious': 0.0}, 
        {'best': 0.07952020911994373, 'pizza': 0.0, 'people': 0.07952020911994373, 'good': 0.029348543175946873, 'loves': 0.0, 'person': 0.0, 'delicious': 0.0}]
    sortedDict = []    
    for d in testList:            
        sortedDict.append(dict( sorted(d.items(), key=lambda x: x[0].lower()) )) 

    expectedOutputDF = pd.DataFrame(sortedDict)

    pd.testing.assert_frame_equal(tfIdf_ftransform.tfIdf_transform(input_str),expectedOutputDF, check_dtype=False)
    boolResult = expectedOutputDF.equals(tfIdf_ftransform.tfIdf_transform(input_str))

    if(boolResult == True):
        return "Pass"
    else:
        return "Fail"    
    
def test_WordClassString(sample1):
    # testing for sample 1. 
    # type(sample1) == str
    expectedTokens1 = ['She', 'is', 'a', 'good', 'person,', 'and', 'she', 'loves', 'pizza@#$%,', "that's", 'probably', 'because', 'of', 'her', 'intestinal^*&', 'prerogatory', 
    'transformation.', 'The', 'neighbours', 'got%£', 'some', 'pizza,', 'enjoying', 'it', 'without', 'electrical', 'assistance..........']
    expectedCount1 = [['She', 1], ['is', 1], ['a', 1], ['good', 1], ['person,', 1], ['and', 1], ['she', 1], ['loves', 1], ['pizza@#$%,', 1], ["that's", 1], ['probably', 1], 
    ['because', 1], ['of', 1], ['her', 1], ['intestinal^*&', 1], ['prerogatory', 1], ['transformation.', 1], ['The', 1], ['neighbours', 1], ['got%£', 1], ['some', 1], ['pizza,', 1], ['enjoying', 1], ['it', 1], ['without', 1], ['electrical', 1], ['assistance..........', 1]]
    expectedStopW1 = [['good', 'person,', 'loves', 'pizza@#$%,', "that's", 'probably', 'intestinal^*&', 'prerogatory', 'transformation.', 'neighbours', 'got%£', 'pizza,', 'enjoying', 'electrical', 'assistance..........']]
    expectedJoinW1 = ["good person, loves pizza@#$%, that's probably intestinal^*& prerogatory transformation. neighbours got%£ pizza, enjoying electrical assistance.........."]
    word = Word(sample1)
    assert word.tokenize() == expectedTokens1
    assert word.word_counter() == expectedCount1
    assert word.remove_stopwords() == expectedStopW1
    assert word.join_stopwords() == expectedJoinW1

def test_WordClassList(sample2):
    # testing for sample 2
    # type(sample2) == list 
    expectedTokens2 = ['This', 'is', 'hell', '&', 'the', 'rest', 'is', 
    'all', 'pizza.', "Tesla's", 'next', 'GigaFactory', 'location', 'may', 'have', 'been', 'revealed.']
    expectedCount2 = [['This', 1], ['is', 2], ['hell', 1], ['&', 1], ['the', 1], ['rest', 1], ['all', 1], ['pizza.', 1], ["Tesla's", 1], 
    ['next', 1], ['GigaFactory', 1], ['location', 1], ['may', 1], ['have', 1], ['been', 1], ['revealed.', 1]]
    expectedStopW2 = [['hell', '&', 'rest', 'pizza.'], ["Tesla's", 'GigaFactory', 'location', 'revealed.']]
    expectedJoinW2 = ['hell & rest pizza.', "Tesla's GigaFactory location revealed."]

    word2 = Word(sample2)
    assert word2.tokenize() == expectedTokens2
    assert word2.word_counter() == expectedCount2
    assert word2.remove_stopwords() == expectedStopW2
    assert word2.join_stopwords() == expectedJoinW2

def test_WordClassHelper(sample1):
    # testing STOPWORD helper functions
    word = [Word(sample1), Word(sample1)]
    stopwords_copy1 = copy(word[0].STOPWORDS)
    stopwords_copy2 = copy(word[1].STOPWORDS)
    extendedWords = stopwords_copy1.append('extended')
    removedWords = stopwords_copy2.remove('afterwards')

    assert word[0].extend_words('extended') == extendedWords
    assert word[1].remove_words('afterwards') == removedWords

if __name__ == "__main__":  
    c = Clean()
    sample = "She is a good person, and she loves pizza@#$%, that's probably because of her intestinal^*& prerogatory transformation. The neighbours got%£ some pizza, enjoying it without electrical assistance.........."
    print('Sample input:', '\t', sample)
    print('\n')
    
    # testing Word Class
    sample2 = ["This is hell & the rest is all pizza.", "Tesla's next GigaFactory location may have been revealed."]
    test_WordClassString(sample)
    test_WordClassList(sample2)
    test_WordClassHelper(sample2)
    print("Testing Word class successful!\n Moving forward to Clean class...")
    
    x = c.remove_symbol(sample)
    print('Removed symbols from input:','\t', x)
    print('\n')
    
    out = c.remove_punctuation(x)
    print('Removed punctations from input:','\t', out)
    print('\n')
    
    pre = c.stem(out)
    print('Removed Stemming:', '\t', pre)
    print('\n')
    
    f = c.join_stopwords(pre)
    print('Removed stop words:', '\t', f)
    print('\n')
   
    ''' below part will be operational only when suddhendras code is complete,
        also input arr will be changed accordingly'''
    # w = Word(c)
    # w = w.join_stopwords()
    # print('Removed stopwords from input --','\t',w) 
    res_vector = count_vector_testing(out)
    if res_vector == None:
        print("Count Vector Method Works, moving forward to Bag of Words")
    result = Bag_of_Words_testing(c)
    if result == None:
        print("Everything is correct")  
        
    TfIdf_fTransform_result = TfIdf_fTransform_testing(out)
    if TfIdf_fTransform_result == "Pass":
        print("Tf-Idf values are correct")
        
    TfIdf_transform_result = TfIdf_transform_testing(out)
    if TfIdf_transform_result == "Pass":
        print("Tf-Idf values are correct")        
