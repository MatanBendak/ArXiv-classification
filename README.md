# ArXiv-classification
Classification task over papers using the titles and abstracts as text

The notebook (Papers Classification using ArXiv.ipynb) was created in kaggle.com notebooks, which has access to the ArXiv meta-data dataset ('/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json') as a JSON file. We wanted to learn a binary classification task between 2 papers categories, in our case we learnt quantitive-finance V.S. Statistics.
In order to prepare the data we needed to load it but it cannot be done in 1 batch. Therefore, we created a generator that yields a paper for every call. We ran over the papers in the dataset and for every paper we extracted the category, after removing the sub-categories. In this way, we could save 1 paper's meta-data at a time to our pandas dataframe using only papers that are in 1 of the two wanted categories. The 2 categories may vary as the user may change the variables _minority_class_ and _majority_class_. A change in those variables will result in loading the relevant data for the wanted categories. When loading a paper, our code extracts some information from the json file, including the category, title, abstract and the update data of the paper.

The notebook contains a class "BinaryPaperClassifierModel". This class can fit 3 types of models: A catboost classifier, a random forest classifier and a Neural Network for binary classification. The fit method in the class runs the preprocessing on the text data which includes removing test that is not in the english language and tokenization, Word2Vec embedding and then it trains the model the user chose, saves the embedding object and model object. Lastly, the class has a "predict" method which uses the saved preprocessing function, the Word2Vec embeddings from the training set and the fitted model - creates a score for each instance.

We trained the 3 models and compared their results. The Neural Network and Catboost had similar results, which both are better than the random forest.
We analyzed the results over time and saw that at about 2018 there was a drop in the performance in all 3 models. We suggested a few root-causes and found the most possible one - the %minority class was changed over time altough we split the data in a stratified way, we didn't do it for each year seperately. We can examine this root-cause by creating a new split to the train-val-test using the year as a group (in addition to the stratification over the label).


![image](https://github.com/MatanBendak/ArXiv-classification/assets/58906443/b06df2b3-aad2-4826-ac6b-b001d273a561)
