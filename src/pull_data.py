import codecs
import constants

class TextTransformer(object):

    from re import sub

    def __init__(self):

        #from nltk.stem.lancaster import LancasterStemmer
        from sklearn.feature_extraction.text import CountVectorizer

        import enchant

        #self.stemmer = LancasterStemmer()
        self._vectorizer = CountVectorizer(strip_accents='ascii')
        self.tokenizer = self._vectorizer.build_tokenizer()
        self.preprocessor = self._vectorizer.build_preprocessor()
        self.spellchecker = enchant.DictWithPWL("en_US",
            pwl=constants.PERSONAL_WORD_DICTIONARY_FILE)


    def transform_text(self, raw_text):
    
        tokens = []
        for token in self.tokenizer(raw_text):
            clean_token = self.preprocessor(token)
            if not self.spellchecker.check(clean_token):
                corrections = self.spellchecker.suggest(clean_token)
                if len(corrections) > 0:
                    clean_token = corrections[0]

            tokens.append(clean_token)

        return ' '.join(tokens)


    def sub_numbers(self, text):
        return sub("[0-9]+", " numbr ", text)


def load_training_data(predefined_size=-1, verbose=False):
    return _load_data(constants.TRAINING_DATA_FILE, predefined_size, verbose=verbose, training_data=True)


def load_testing_data(verbose=False):
    return _load_data(constants.TESTING_DATA_FILE, verbose=verbose, training_data=False)


def _load_data(data_file, predefined_size=-1, verbose=False, training_data=True):

    if verbose: print "Loading data from: %s" % data_file

    data = list()
    transformer = TextTransformer()
    
    if training_data:
        labels = list()
    else:
        labels = None

    def split_date_and_add_type_strings(raw_string, identifier):
        ''' 
            Break up a date string and label each chunk
            separately so that we can use each chunk as separate
            features. 
            Input:
                * an identifying label to attach to each chunk
                * the raw text to chunkify
            Output:
                * a string of individual chunks
        '''
        year, month, day = raw_string.split('-')

        year += '_' + identifier + '_year'
        month += '_' + identifier + '_month'
        day += '_' + identifier + '_day'

        return year + ' ' + month + ' ' + day

    # read in data.
    with codecs.open(data_file, encoding='utf-8', mode='rb') as fh:

        fh.next()   # eat the header.

        for row in fh:

            if training_data:
                _, sku, category, query, click_timestamp, query_timestamp = row.split(',')
            else:
                _, category, query, click_timestamp, query_timestamp = row.split(',')

            if predefined_size != -1 and len(data) > predefined_size:
                break

            if training_data:
                labels.append(int(sku))

            datum = category + ' ' + transformer.transform_text(query)

            click_date, click_time = click_timestamp.split()

            datum += ' ' + split_date_and_add_type_strings(click_date, 'click')

            query_date, query_time = query_timestamp.split()

            datum += ' ' + split_date_and_add_type_strings(query_date, 'query')

            data.append(datum)

    if training_data:
        return data, labels
    else:
        return data
