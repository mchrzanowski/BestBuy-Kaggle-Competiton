BestBuy-Kaggle-Competiton
=========================

BestBuy Kaggle Small Data Contest Attempt Using One-Versus-One Multiclass Classification 
Kaggle Competiton: http://www.kaggle.com/c/acm-sf-chapter-hackathon-small

Dependencies:
	* bs4
	* numpy
	* scipy
	* sklearn
	* pyenchant

To train the classifier, run the following command from the src directory:
	* python train_classifier.py --store

To generate the predictions file, which will be output in ./etc/, run the following command from the src directory:
	* python predict.py
