# Sentiment-Analysis-in-Yelp-Review-Dataset

This project was developed for the EECS 6412 Data Mining course at York University.  

Initially, we preprocessed the yelp review dataset which was originally in the .json format. 

We selected 100000 reviews randomly from the original review dataset of Yelp. Our selected dataset is balanced with each star rating (1 to 5 star) contains 20000 reviews.  

Our selected pre-processed datasets are in *.txt* format and can be found in the *"datasets.zip"* folder. 

We have two types of datasets:<br />
<pre>
    i. .txt file for Python (Separate file for binary and ternary classifications)<br />
    ii. .arff file for Weka (Separate file for binary and ternary classifications)<br />
</pre>

The file size of the original *.json* file is 4.7 GB. The .json file can be downloaded from here: (https://www.yelp.com/dataset)  

All the Deep Learning Algorithms are implemented in Python using Keras Library.  

Multinomial Naive Bayes is implemented using Scikit-learn library in Python.  

SVM and Random Forest are implemented in WEKA using filtered classifier. A tutorial video regarding how to use the filtered classifier in WEKA can be found here: (https://www.youtube.com/watch?v=PJ-vSXHoXYM). 

See the file "User's Manual.pdf" for details.

You can read this awesome tutorial regarding Keras Model for Neural Network: (https://mc.ai/using-word-embeddings-and-recurrent-neural-networks-to-predict-rating-scores-from-text/)
