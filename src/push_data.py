import constants

def write_prediction_file(predictions, verbose=False):

    if verbose:
        print "Writing predictions to: %s" % constants.TESTING_PREDICTIONS_FILE

    with open(constants.TESTING_PREDICTIONS_FILE, 'wb') as prediction_file:
        for row in predictions:
            prediction_file.write(' '.join(str(value) for value in row) + '\n')

    if verbose:
        print "Writing Finished."
