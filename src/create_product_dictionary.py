from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup

import codecs

def run():
    ''' create a product dictionary based on all tokens in the best buy product corpus '''

    soup = BeautifulSoup(open(constants.BESTBUY_PRODUCT_CORPUS_FILE, 'rb'), 'html.parser')
    vectorizer = CountVectorizer(strip_accents='ascii')
    
    tokenizer = vectorizer.build_tokenizer()
    preprocessor = vectorizer.build_preprocessor()

    tokens = set()

    for item in tokenizer(soup.get_text()):
        tokens.add(preprocessor(item))

    with codecs.open(constants.PERSONAL_WORD_DICTIONARY_FILE, mode='wb', encoding='utf-8') as f:
        for token in tokens:
            f.write(token + '\n')


if __name__ == '__main__':
    import time
    start = time.time()
    run()
    end = time.time()
    print "Runtime: %f seconds" % (end - start)
