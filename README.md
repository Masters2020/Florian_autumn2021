This repository includes all the code used for the Thesis 'Using entity-action-target relationships to classify conspiratorial youtube videos'.

To summarize, the thesis used a conspiracy corpus, in this case conspiratorial reddit forums to extract relations between mentioned entities actions they perform and the target of the action. These triplets are then used to compare them to similar relations in youtube videos to determine if they are conspiratorial.

The code is split into 11 scripts, originally Jupyter Notebooks, following the different steps of the pipeline, a short explanation of each script is given below.

Additonally some of the resulting files from the code are uploaded as well and are also explained how they relate to the notebook and what they do. Intermediary results were not uploaded to avoid confusion regarding the amount of files.

## **Scripts**

01_reddit scrapping: this script implements the reddit scrapping api to scrape the different forums. It requires a reddit account and more info can be found here: https://www.storybench.org/how-to-scrape-reddit-with-python/

02_preprocessing: this script preprocesses the scrapped reddit posts and also the youtube video transcripts for further use.

03_EDA: this script includes a few lines of code to explore the data, e.g. the label distribution of the youtube videos etc. It is not really relevant for the further pipeline.

04_ner: this script performs an NER task on both datasets which are used as a basis for the fine-tuning of the BERT models (the entities are added to the BERT vocab to prevent the splitting into sub tokens in script **05** and also are the base for the triplets in script **08**.

05_fine_tuning BERT: this script creates a BERT model fine-tuned on a specified dataset. Those models are then used to extract the embeddings from the datasets. The script or notebook was implemented in google collab since it uses cuda and cuda did originally not work on locally on my machine.

06_embeddings: this script extracts the embeddings for the datasets. It uses the BERT model from script **05**. It takes a lot of time to run, therefore some of the embeddings can be found here: 

07_preparing triplets: this script prepares the data for the final combination of triplets, it does such thing as getting all the verbs or nouns and their positions in an instances of a datasets, etc. The resulting files are used for script **08**.

08_triplets: based on the embeddings from script **06** and the preparatory files from script **07** this script creates the triplets for the datasets.

09_splitting jsons: this script splits the resulting files from script **08** according to training and testing. Therefore this is only really relevant if a dataset need to be split into training and testing to classify the testing set utilising the training set with the method used in these scripts.

10_cleaning jsons: this script does some rudimentary clean-up of the jsons in case of any problems. Generally this should not be needed, since most problems that were fixed with this script were fixed in the original code.

11_classification: this script uses the files from script **08**, **09** or **10** to create the classification results by using a lookup mechanism between the dataset (reddit) used to classify another dataset (youtube).

