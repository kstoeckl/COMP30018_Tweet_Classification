**Overview:**
    A project for COMP30018 Knowledge Technologies which I later continued
    in my own time.

    Given a collection of Tweets of the form,

    UserID  TweetID Message                      LocationTag

    756546  5362058 Listeing to Mac Break Weekly    SD

    We were tasked with predicting the location of Twitter User based upon the
    content of their tweet.

    The possible locations of the users and the corresponding tags were,
    
        | Boston          B
        | Seatle          Se
        | Houston         H
        | Washington      W
        | San Diego       SD


    We had 3 sets of Tweets, a training set, a development/validation set
    and a testing set.

**Process:**
    **Pre-Processing:**
    First the tweets were processed using regex, both making the format of
    the tweet more consistent and also retrieving key pieces of data such
    as the UserID and Location Tags.

    A vector space model was employed to encode the message of the tweet,

    e.g. for instance x, x[i] = j <=> there are j lots of word i in x.

    Sparse Matrices were used to encode this efficiently, the corresponding
    y value for each instance was simply left as the location tag.

    **Feature Selection:**

    The Chi2 test was then used to identify statistically significant features
    in the training data.

        *A number of feature refinements were also trialled in this stage
        for instance, stemming and expanding the vector space to include 
        ngrams, for instance the phrase "two words" would be stored as one
        word in the vector space. However, neither were found to improve 
        performance.

    **Data Formatting:**

    The Training, Validation and Test data were encoded using the vector
    space model, which had been refined to only include the 'useful' features.

    A key step in this process was aggregating tweets which had the
    same UserID in the Validation and Test data sets. This enabled the 
    classifier to assign locations on a per user basis, which it was able to 
    do significantly more accurately thanks to the availability of more data.

    **Training and Classification:**

    Two models were trialled throughout the project both from the sklearn
    library, a Multinomial Naive Bayes Classifier and the SGDClassifier 
    (Used to fit a linear Support Vector Machine through Stochastic Gradient 
    Descent).

    The parameters of the Classifiers were fine-tuned through the use of
    hyper-parameter optimization.

**Results:**
    The best performing classifier found was:

    SGDClassifier

    Parameters:
        | n_features=150000
        | length_ngram=1
        | OPT_ALPHA=0.0001
        | L1_RATIO = 0.3333
        | N_ITER = 50

    Accuracy Validation Tweets:
        70.43%
    Accuracy Test Tweets:
        68.54%

    The following additional metrics and graphics correspond to the 
    Validation Tweets.

    ..figure:: https://raw.githubusercontent.com/kstoeckl/COMP30018_Tweet_Classification/master/ConfusionMatrix.png

**Extensions:**
    There are a number of possible extensions too this project.

    A lot of merit lies in feature engineering, although my attempts to 
    expand the vector space to include phrases (varying length_ngram)
    and conducting stemming were fruitless, it is quite likely that a 
    more prolonged attempt could prove successful.
    
    Other possible extensions could be to run the tweet through a 
    spell-checker, group tweets based on language or perhaps translate
    all tweets into English (many tweets were in Spanish).

    Another area for improvement lies in the Classifier itself.

    Other types of Classifiers could be trialled and the process of 
    hyper-parameter optimization could be improved further. Aside
    from using an alternative (better) optimization algorithm, results could
    also be improved by expanding the parameter search space, both with regard
    to 'density' of the search being done and also the parameters (For
    instance you could tinker with the learning rate of the SGDClassifer).


**Usage:**

performanceTest.py:
    Main Script used for conducting trials varying the size of the ngrams
    and the number of features. Makes use of the other python scripts.
    Also outputs the final test results.

All of the below python scripts may be run separately, in which case their
output is saved through the use of pickle and then used by the next
script. (Slower in total than running performanceTest.py due to not all
data structures being saved, however modularity allows the components to
be tested individually and the Classifier to be trained and tested without
recreating the vector space model, the most time intensive step.)

featureSelection.py:
    Uses Regex, countVectorizer and the Chi2 Test to identify the k 
    best ngrams in terms of their score in the Chi2 test.

    Significant code segments modified from `here
    <http://scikit-learn.org/dev/auto_examples/text/document_classification_20newsgroups.html#>`_.


formatData.py:
    Takes a given set of feature names, the set of training tweets and
    the set of testing/validation tweets and constructs Xtrain, Ytrain,
    Xtest and Ytest.
model.py:
    Constructs the model, trains it on the training data
    and tests it on the validation data. 

    Also has the capacity to hyper-parameter optimization, however this code 
    is currently commented out.
    Also generates the confusion_matrix graphic using code modified from `here
    <http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py>`_.


**Tweet Files:**

Samples of the tweet txt files have also been included.