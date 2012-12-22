import constants
import numpy
import pickle
import pull_data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


def learn(training_data, training_labels, show_score=False, store=False, verbose=False):

    if verbose:
        print "Start Learning...."
    
    clf = SVC(kernel='linear', probability=True, C=2)

    clf.fit(training_data, training_labels)

    if verbose:
        print "Done Learning."

    if store:
        if verbose:
            print "Pickling classifier..."
        pickle.dump(clf, open(constants.CLASSIFIER_PICKLING_FILE, 'wb'))
        if verbose:
            print "Done Pickling."

    # do this after pickling as it takes a while; we can run predict.py
    # while the score is computing.
    if show_score:
        if verbose:
            print "Scoring classifier ..."
        print "Data-Level Training Set Prediction Accuracy: %s" % \
            clf.score(training_data, training_labels)
        if verbose:
            print "Scoring Finished."


def construct_tf_idf_matrix(data, store=False, verbose=False):

    if verbose:
        print "TF-IDF Normalized Matrix Construction..."

    # create tf-idf-normalized training matrix.
    vectorizer = TfidfVectorizer(stop_words='english', charset_error='ignore')
    training_data = vectorizer.fit_transform(data)

    if verbose:
        print "Done Constructing Matrix"

    if store:
        if verbose:
            print "Pickling Trained Transformer..."
        pickle.dump(vectorizer, open(constants.TRANSFORMER_PICKLING_FILE, 'wb'))
        if verbose:
            print "Pickling Done."

    return training_data


def create_label_mapping(labels, store=False, verbose=False):
    '''
        The SKUs are normally enormous numbers, and the classifiers (e.g., SVC) actually
        internally store labels as signed int32 numbers. This creates a problem when
        using predict_proba(X) as that will output class probabilities based on the 
        arithmetic order of the stored labels.
        So, map the SKUs to smaller numbers that we then use for classification. 
        Store a reverse mapping for later use in making predictions.
        Input:
            * a list of raw SKUs
        Output:
            * an array of mapped labels.
    '''

    mapping = dict()
    reverse_mapping = dict()

    for i, value in enumerate(set(labels)):
        mapping[value] = i
        reverse_mapping[i] = value

    new_labels = numpy.zeros(shape=len(labels), dtype=numpy.int)
    for i, value in enumerate(labels):
        new_labels[i] = mapping[value]

    if store:
        if verbose:
            print "Pickling Label to SKU Mapping..."
            pickle.dump(reverse_mapping, open(
                constants.LABEL_TO_SKU_MAPPING_PICKLING_FILE, 'wb'))
        if verbose:
            print "Pickling Done."

    return new_labels


def run(store=False, predefined_size=-1, show_score=False, verbose=False, run_diagnostics=False):

    data, labels = pull_data.load_training_data(predefined_size, verbose=verbose)
    print "Samples: %d" % len(data)

    training_labels = create_label_mapping(labels, store, verbose)

    training_data = construct_tf_idf_matrix(data, store, verbose)
    print "Training Matrix size: %s x %s" % training_data.shape

    if run_diagnostics:
        import diagnostics
        diagnostics.find_best_parameters_for_SVM(training_data, training_labels, verbose)
    else:
        learn(training_data, training_labels, show_score, store, verbose)


if __name__ == '__main__':
    import argparse
    import time
    
    start = time.time()
    parser = argparse.ArgumentParser(description="""Train a multi-class
        classifier on BestBuy data that can then be used to output 
        the most likely labels for a sample.""")

    parser.add_argument('--store', action='store_true', default=False,
        help='Pickle important objects to do prediction later.')
    parser.add_argument('--verbose', action='store_true', default=False,
        help='Verbose mode.')
    parser.add_argument('-size', type=int, default=-1,
        help='Predefined training set size.')
    parser.add_argument('--show_score', action='store_true', default=False,
        help='Score the accuracy of the classifier (this takes a while).')
    parser.add_argument('--diagnostics', action='store_true', default=False,
        help='Run diagnostics instead of training a classifier.')

    args = vars(parser.parse_args())
    run(args['store'], args['size'], args['show_score'], args['verbose'], args['diagnostics'])

    end = time.time()
    print "Runtime: %f seconds" % (end - start)
