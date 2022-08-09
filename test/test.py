from project import * 

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
if __name__ == "__main__":  
    
    c = Clean()
    x = c.remove_symbol(['She loves pizza&%^$, pizza is ^*&delicious',
                              'She is a good pers%£%^on',
                              'good people%..... are%£ the best are'])
    print('Removed Symbol from input --','\t',x)
    print('\n')
    out = c.remove_punctuation(['She loves pizza, pizza is delicious',
                              'She is a good person',
                              'good people..... are the best are'])
    print('Removed Punctation from input --','\t',out)
    print('\n')   
    ''' below part will be operational only when suddhendras code is complete,
        also input arr will be changed accordingly'''
    # w = Word(c)
    # w = w.join_stopwords()
    # print('Removed stopwords from input --','\t',w) 
    res_vector = count_vector_testing(out)
    if result == None:
        print("Count Vector Method Works, moving forward to Bag of Words")
    result = Bag_of_Words_testing(c)
    if result == None:
        print("Everything is correct")  
