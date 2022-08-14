# Husky_Simplex
Text processing package

Data preprocessing is the first and most essential stage in developing a machine learning
model as it affects the overall accuracy and efficiency of the outcome. Ordinary text data
contains non-contextual words, noise, misspelled words, symbols, punctuations, and
unnecessary syntactic connotations. To circumvent these hindrances, we need to clean
raw text data into data that is acceptable for statistical and computational analysis.

The purpose of the package is to provide a one-stop platform for most of the necessary
text preprocessing techniques. These steps are used to augment the computational
significance of text data for Natural Language Processing tasks.

## Package Functions
We have implemented three classes: Word, Clean, and Vectorizer. The Word class would contain methods that deal with words in the text data like Tokenization, Word Counter, and Stopword removal. Next, the Clean class deals with correcting noise and non-contextual words with no statistical significance. Punctuation removal, Symbol removal, Sentence splitting, and Stemming are the methods included in this class. And finally, the Vectorizer class contains the Bag of Words, Count_vectorizer, and the TFIDF_vectorizer methods. Overall, we will be implementing methods with the following functionalities:
1. Tokenization - Converting string input to a list of words.
2. Word counter - Counting the total number of words in the input.
3. Stopword removal - Removing non-contextual words that are only used for the
grammatical structure.
4. Punctuation removal - Removing punctuations.
5. Symbol removal - Removing symbols.
6. Stemming - Removing tense connotations.
7. Bag of words - Quantifying words.
8. Count vectorization - Vectorization of text based on term frequency.
9. TF-IDF vectorization - Vectorization of text based on term frequency in relation to
document frequency

## Installation
``` pip install husky_simplex ```

or <br />
``` git clone https://github.com/Sudhendra/Husky_Simplex.git ```<br />
``` cd Husky_Simplex ```<br />
``` pip install - r requirements.txt ```
