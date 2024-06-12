import datetime
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
import nltk
from nltk.stem import SnowballStemmer
import numpy as np
import re
from tabulate import tabulate
import evaluate
from sklearn.utils import compute_class_weight

# %%
nltk.download("stopwords")

# %% Global settings

number_removal = True
apply_stemming = True
apply_stopwords = True
vectorization = 'count'
text_field = "concat_text"
prediction_field = "post"


class NaiveBayesClassifier:
    def __init__(self, dataset_test, dataset_train, number_removal=True, apply_stemming=True, apply_stopwords=True,
                 vectorization='count', save_false_predictions=False, model="multinomial", test_200=True):
        self.model = ComplementNB() if model == "complementNB" else MultinomialNB()
        self.df_train = pd.read_csv(dataset_train)
        self.df_test = pd.read_csv(dataset_test)

        if test_200:
            self.df_200= pd.read_csv("data/pipeline_runs/classification/annotated_200_only_night.csv")

        self.accuracy = evaluate.load("accuracy")
        self.precision = evaluate.load("precision")
        self.recall = evaluate.load("recall")
        self.f1 = evaluate.load("f1")

        self.dataset_train = dataset_train
        self.number_removal = number_removal
        self.apply_stemming = apply_stemming
        self.apply_stopwords = apply_stopwords
        self.vectorization = vectorization
        self.stopwords = list(stopwords.words('norwegian')) if apply_stopwords else None
        self.stemmer = SnowballStemmer("norwegian")
        self.save_false_predictions = save_false_predictions

    def pre_processing(self):
        self.df_train[text_field] = self.df_train[text_field].str.lower()
        self.df_test[text_field] = self.df_test[text_field].str.lower()
        self.df_200[text_field] = self.df_200[text_field].str.lower()

        if self.number_removal:
            self.df_train[text_field] = self.df_train[text_field].apply(lambda x: re.sub(r'\d+', '', x))
            self.df_test[text_field] = self.df_test[text_field].apply(lambda x: re.sub(r'\d+', '', x))
            self.df_200[text_field] = self.df_200[text_field].apply(lambda x: re.sub(r'\d+', '', x))

        if self.apply_stemming:
            self.df_train[text_field] = self.df_train[text_field].apply(
                lambda x: ' '.join([self.stemmer.stem(token) for token in x.split()]))
            self.df_test[text_field] = self.df_test[text_field].apply(
                lambda x: ' '.join([self.stemmer.stem(token) for token in x.split()]))
            self.df_200[text_field] = self.df_200[text_field].apply(
                lambda x: ' '.join([self.stemmer.stem(token) for token in x.split()]))

    def run_model(self):
        # Split into train and test datasets
        # X_train, X_test, y_train, y_test = train_test_split(self.df['text'], self.df['category'], test_size=0.2, random_state=42)

        X_train = self.df_train[text_field]
        X_test = self.df_test[text_field]
        y_train = self.df_train[prediction_field]
        y_test = self.df_test[prediction_field]

        day_test = self.df_200[text_field]
        night_test = self.df_200[text_field]
        day_y = self.df_200['Dag']
        night_y = self.df_200['Natt']


        # Apply Naive Bayes
        if self.vectorization == 'count':
            model = make_pipeline(CountVectorizer(ngram_range=(1, 3), stop_words=self.stopwords), self.model)
        else:
            model = make_pipeline(TfidfVectorizer(stop_words=self.stopwords), self.model)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        day_pred = model.predict(day_test)
        night_pred = model.predict(night_test)

        accuracy = self.accuracy.compute(references=y_test, predictions=y_pred)['accuracy']
        precision = self.precision.compute(references=y_test, predictions=y_pred)['precision']
        recall = self.recall.compute(references=y_test, predictions=y_pred)['recall']
        f1 = self.f1.compute(references=y_test, predictions=y_pred)['f1']


        day_accuracy = self.accuracy.compute(references=day_y, predictions=day_pred)['accuracy']
        day_precision = self.precision.compute(references=day_y, predictions=day_pred)['precision']
        day_recall = self.recall.compute(references=day_y, predictions=day_pred)['recall']
        day_f1 = self.f1.compute(references=day_y, predictions=day_pred)['f1']

        night_accuracy = self.accuracy.compute(references=night_y, predictions=night_pred)['accuracy']
        night_precision = self.precision.compute(references=night_y, predictions=night_pred)['precision']
        night_recall = self.recall.compute(references=night_y, predictions=night_pred)['recall']
        night_f1 = self.f1.compute(references=night_y, predictions=night_pred)['f1']


        result_list.append(
            {
                "Model name": self.model,
                "Note": category + self.dataset_train,
                "Number removal": self.number_removal,
                "Stemming": self.apply_stemming,
                "Stopwords": self.apply_stopwords,
                "Vectorization": self.vectorization,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "day_accuracy": day_accuracy,
                "day_precision": day_precision,
                "day_recall": day_recall,
                "day_f1": day_f1,
                "night_accuracy": night_accuracy,
                "night_precision": night_precision,
                "night_recall": night_recall,
                "night_f1": night_f1,
            }
        )

        report = classification_report(y_test, y_pred)
        # print(report)

        self.print_scores(accuracy, precision, recall, f1)

        # Find the incorrect predictions
        incorrect_entries = [index for index, (true, pred) in enumerate(zip(y_test, y_pred)) if true != pred]
        self.df_test.iloc[incorrect_entries].to_csv("incorrect_naive.csv")
        incorrect_texts = X_test.iloc[incorrect_entries].tolist()
        true_labels = y_test.iloc[incorrect_entries].tolist()
        pred_labels = [y_pred[index] for index in incorrect_entries]

        if self.save_false_predictions:
            self.incorrect_texts(incorrect_texts, true_labels, pred_labels)

    def print_scores(self, accuracy, precision, recall, f1):
        # number_removal=True, apply_stemming=True, apply_stopwords=True, vectorization='bow'
        variables = ["Model", "Train Dataset", "Number Removal", "Stemming", "Stopwords", "Vectorization", "Accuracy",
                     "Precision", "Recall", "F1"]
        values = [
            [self.model, category + self.dataset_train, self.number_removal, self.apply_stemming, self.apply_stopwords,
             self.vectorization,
             accuracy, precision, recall, f1]]

        print(tabulate(values, headers=variables, tablefmt='grid'))

    def incorrect_texts(self, incorrect_texts, true_labels, pred_labels):
        df_misclassified = pd.DataFrame({
            'Text': incorrect_texts,
            'True Label ID': true_labels,
            'Predicted Label ID': pred_labels,
        })

        folder_name = f'data/pipeline_runs/classification/naive_classification - d:{datetime.datetime.now().day} m:{datetime.datetime.now().month} h:{datetime.datetime.now().hour}/'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        # Save the DataFrame to a CSV file
        df_misclassified.to_csv(f'{folder_name}misclassified_texts-{self.vectorization}.csv', index=False)
        print("Misclassified texts saved to 'misclassified_texts.csv'")

    def main(self):
        self.pre_processing()
        self.run_model()


def run_combination(file_name, model=""):
    classifier = NaiveBayesClassifier(
        test_file,
        file_name,
        number_removal=False, apply_stemming=True,
        apply_stopwords=True, vectorization='count', model=model)
    classifier.main()

    classifier = NaiveBayesClassifier(
        test_file,
        file_name,
        number_removal=True, apply_stemming=True,
        apply_stopwords=True, vectorization='count', model=model)
    classifier.main()

    classifier = NaiveBayesClassifier(
        test_file,
        file_name,
        number_removal=False, apply_stemming=True,
        apply_stopwords=True, vectorization='tf-idf', model=model)
    classifier.main()

    classifier = NaiveBayesClassifier(
        test_file,
        file_name,
        number_removal=True, apply_stemming=True,
        apply_stopwords=True, vectorization='tf-idf', model=model)
    classifier.main()

    print("")


if __name__ == '__main__':
    result_list = []
    category = "without_night"
    path_to_data = f"data/pipeline_runs/classification/{category}/"
    test_file = f"{path_to_data}static_test.csv"

    sampling_type = "100k_as_negative.csv"
    night_100k = f"{path_to_data}{sampling_type}"

    sampling_type = "after_2020_rest_90_percent.csv"
    rest_90 = f"{path_to_data}{sampling_type}"

    sampling_type = "all_except_test.csv"
    all_except = f"{path_to_data}{sampling_type}"

    run_combination(night_100k)
    run_combination(night_100k, model="complementNB")

    run_combination(rest_90)
    run_combination(rest_90, model="complementNB")

    run_combination(all_except)
    run_combination(all_except, model="complementNB")

    category = "with_night"
    path_to_data = f"data/pipeline_runs/classification/{category}/"
    test_file = f"{path_to_data}static_test.csv"

    sampling_type = "100k_as_negative.csv"
    night_100k = f"{path_to_data}{sampling_type}"

    sampling_type = "after_2020_rest_90_percent.csv"
    rest_90 = f"{path_to_data}{sampling_type}"

    sampling_type = "all_except_test.csv"
    all_except = f"{path_to_data}{sampling_type}"

    run_combination(night_100k)
    run_combination(night_100k, model="complementNB")

    run_combination(rest_90)
    run_combination(rest_90, model="complementNB")

    run_combination(all_except)
    run_combination(all_except, model="complementNB")

    res_df = pd.DataFrame(
        result_list)

    print(pd.DataFrame(
        result_list).to_latex(
        escape=True))
