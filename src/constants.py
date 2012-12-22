import os.path

TRAINING_TEST_SPLIT = 0.8

# assuming this is the src folder
BASE_DIR = os.path.curdir

ETC_DIR = os.path.join(BASE_DIR, "../etc")
BESTBUY_PRODUCT_CORPUS_FILE = os.path.join(ETC_DIR, "small_product_data.xml")
TRAINING_DATA_FILE = os.path.join(ETC_DIR, "train.csv")
TESTING_DATA_FILE = os.path.join(ETC_DIR, "test.csv")
TESTING_PREDICTIONS_FILE = os.path.join(ETC_DIR, "predictions.csv")
PERSONAL_WORD_DICTIONARY_FILE = os.path.join(ETC_DIR, "product_dictionary.txt")

PICKLING_DIR = os.path.join(BASE_DIR, "../pickled")
CLASSIFIER_PICKLING_FILE = os.path.join(PICKLING_DIR, "classifier")
LABEL_TO_SKU_MAPPING_PICKLING_FILE = os.path.join(PICKLING_DIR, "label_to_sku_mapping")
TRANSFORMER_PICKLING_FILE = os.path.join(PICKLING_DIR, "tfidf_transformer")
