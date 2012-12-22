import constants
import numpy
import pickle
import pull_data
import push_data


def make_predictions(data, number_of_predictions, verbose=False):
    if verbose:
        print "Classifying data using %s" % constants.CLASSIFIER_PICKLING_FILE

    classifier = pickle.load(open(constants.CLASSIFIER_PICKLING_FILE, 'rb'))

    if verbose:
        print "Assuming I need to use the label-To-SKU mapper located at: %s" \
            % constants.LABEL_TO_SKU_MAPPING_PICKLING_FILE

    label_to_sku_mapping = pickle.load(open(constants.LABEL_TO_SKU_MAPPING_PICKLING_FILE, 'rb'))

    if number_of_predictions > len(label_to_sku_mapping):
        raise Exception("""Too many predictions requested.
            \nRequested: %s\nAvailable: %s """ % (number_of_predictions,
            len(label_to_sku_mapping)))

    predictions = classifier.predict_proba(data)

    predictions_to_return = numpy.zeros(shape=(data.shape[0],
        number_of_predictions), dtype=numpy.uint64)

    for i in xrange(predictions.shape[0]):
        values_to_indices = dict( ( (value, j) for j, value in enumerate(predictions[i]) ) )
        sorted_keys = sorted(values_to_indices.keys(), reverse=True)

        for j in xrange(number_of_predictions):
            prediction = values_to_indices[sorted_keys[j]]
            predictions_to_return[i, j] = label_to_sku_mapping[prediction]

    return predictions_to_return


def construct_tf_idf_matrix(data, verbose=False):

    if verbose:
        print "Transforming data using %s" % constants.TRANSFORMER_PICKLING_FILE

    transformer = pickle.load(open(constants.TRANSFORMER_PICKLING_FILE, 'rb'))

    transformed_data = transformer.transform(data)

    if verbose:
        print "Transformation Complete"

    return transformed_data


def run(number_of_predictions, verbose=False):
    if verbose:
        print "Number of Predictions: %s" % number_of_predictions

    raw_testing_data = pull_data.load_testing_data(verbose=verbose)
    print "Samples: %s" % len(raw_testing_data)

    testing_data = construct_tf_idf_matrix(raw_testing_data, verbose)
    print "Testing Matrix size: %s x %s" % testing_data.shape

    prediction_data = make_predictions(testing_data, number_of_predictions, verbose)

    push_data.write_prediction_file(prediction_data)


if __name__ == '__main__':
    import time
    start = time.time()

    import argparse
    parser = argparse.ArgumentParser(description="""Predict the most likely labels
        for each sample in the testing set. """)

    parser.add_argument('-predictions', type=int, default=5,
        help='Number of predictions to make.')

    parser.add_argument("--verbose", action='store_true', default=False,
        help='Verbose mode.')

    args = vars(parser.parse_args())
    run(args['predictions'], args['verbose'])

    end = time.time()
    print "Runtime: %f seconds" % (end - start)
