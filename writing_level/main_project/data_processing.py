#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code contains all the necessary functions for data preprocessing and
feeding the final vectorised data to the model
@author: rtwik
"""
import pandas as pd
import numpy
from collections import Counter
import enchant
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class data_preprocess(object):

    def __init__(self):
        pass

    def get_spelling_mistakes(self, text):
        # claculates the number of spelling mistakes in text
        spell_check = enchant.Dict('en-GB')

        mistakes = 0
        for word in text.split():
            if not spell_check.check(word):
                mistakes += 1

        return mistakes

    def get_word_frequency_from_file(self, filepath):
        # fetches word occurrence frequency in wiki corpus from file
        word_frequency = pd.read_csv(filepath, comment='#',
                                     header=None)
        word_frequency.drop_duplicates(inplace=True)

        return word_frequency

    def get_stopwords(self):
        # returns a list of stopwords
        return set(stopwords.words('english'))

    def get_words_rank_value(self, word_counts, word_frequency):
        # claculates frequency based ranks of words in a given text
        lemma = WordNetLemmatizer()
        stops = self.get_stopwords()

        ranks = []
        for word in list(word_counts.keys()):
            if lemma.lemmatize(word) in list(word_frequency[0])  and word not in stops:

                rank = word_frequency.index[
                        word_frequency[0] == lemma.lemmatize(word)].tolist()

                ranks.append(rank[0]+1)

        ranks.sort()

        return numpy.log(numpy.sum(ranks[-5:]))

    def get_entropy(self, counts):
        # calculates the information entropy of the text
        total_count = numpy.sum(list(counts.values()))
        entropy = numpy.sum([-(i/total_count)*numpy.log2(i/total_count)
                            for i in counts.values()])

        return entropy

    def get_features(self, chunk, word_frequency):
        # collates all the input features for a given text
        n_words = []
        n_unique_words = []
        median_word_length = []
        entropy = []

        for text in chunk['text']:
            text = re.sub("[^a-zA-Z0-9]", " ", str(text))
            counts = Counter(text.split())

            n_word = numpy.sum(list(counts.values()))
            n_words.append(n_word)

            n_unique_words.append(len(counts))

            median_word_length.append(numpy.median(
                [len(words) for words in text.split()]))

            entropy.append(self.get_entropy(counts))

        features = {'n_words': n_words, 'n_unique_words': n_unique_words,
                    'median': median_word_length, 'entropy': entropy}

        return features

    def get_features_extra(self, chunk, word_frequency):
        # has two additional features compared to "get_features", n_mistakes
        # and rank_vlaues. These take very long to calculate. use if needed
        n_words = []
        n_unique_words = []
        median_word_length = []
        entropy = []
        n_mistakes = []
        rank_values = []

        for text in chunk['text']:
            text = re.sub("[^a-zA-Z0-9]", " ", str(text))
            counts = Counter(text.split())

            n_word = numpy.sum(list(counts.values()))
            n_words.append(n_word)

            n_unique_words.append(len(counts))

            median_word_length.append(numpy.median(
                [len(words) for words in text.split()]))

            entropy.append(self.get_entropy(counts))

            n_mistakes.append(self.get_spelling_mistakes(text)/n_word)

            rank_values.append(self.get_words_rank_value(counts,
                                                         word_frequency))

        features = {'n_words': n_words, 'n_unique_words': n_unique_words,
                    'median': median_word_length, 'entropy': entropy,
                    'n_mistakes': n_mistakes, 'rank_values': rank_values}

        return features

    def get_levels(self, chunk):
        # extracts the levels of the texts
        levels = []
        for level in chunk['@level']:
            levels.append(level)

        return {'levels': levels}

    def vectorise_lists_dict(self, lists_dict, batch_size):
        # makes vectors of the features
        vectors = {}
        for f in lists_dict:
            vectors[f] = numpy.array(lists_dict[f]).reshape(batch_size, 1)

        return vectors

    def compile_inputs(self, vector_dict):
        # prepates inputs as arrays needed for model functions
        x = [vector_dict[i] for i in vector_dict]
        x = numpy.hstack(x)

        return x

    def compile_outputs(self, vector_dict, output_classes, output_map):
        # prepates outputs as arrays needed for model functions
        outputs = []

        for l in vector_dict:
            for v in vector_dict[l]:
                y = numpy.zeros(len(output_classes))
                y[output_map[v[0]]-1] = 1
                outputs.append(y)

        return numpy.array(outputs)

    def get_data_length(self, filepath):
        # gets the number of datapoints in dataframe
        df = pd.read_csv(filepath, index_col=0)
        length = len(df)
        del(df)
        return length

    def get_output_map(self, output_list, group_by):
        # map from level number to target value, if groupby=1 then produces
        # a regular map, if groupby=n, maps groups of n consecutive levels
        # to one target value. groupby=3 for task 2
        output_map = {}
        for i in range(int(len(output_list)/group_by)+1):
            groups = output_list[i * group_by: (i+1) * group_by]
            for g in groups:
                output_map[g] = i+1

        return output_map

    def get_output_classes(self, filepath, group_by):
        # lists unique output classes from data
        df = pd.read_csv(filepath, index_col=0)

        output_list = list(set(df['@level']))

        output_map = self.get_output_map(output_list, group_by)
        output_classes = set([output_map[i] for i in output_list])

        del(df)

        return list(output_classes)

    def get_targets_for_eval(self, data_gen, n_chunks):
        # prepares output target values in a way required for F score metrics
        targets = []
        count = 0
        for i in data_gen:
            targets.append(i[1])
            count += 1

            if count == n_chunks:
                break
        targets = numpy.vstack(targets)

        return targets


class data_gen(data_preprocess):
    # iterator object to stream data to the model
    def __init__(self, filepath, params, n_chunks, word_frequency):
        data_preprocess.__init__(self)
        self.params = params
        self.batch_size = params['batch_size']
        self.filepath = filepath
        self.word_frequency = word_frequency
        self.n_chunks = n_chunks
        self.output_map = params['output_map']

    def __iter__(self):
        return self

    def __next__(self):
        chunk_count = 0
        for chunk in pd.read_csv(self.filepath,
                                 chunksize=self.batch_size, index_col=0):
            features = self.get_features(chunk, self.word_frequency)
            feature_vectors = self.vectorise_lists_dict(features,
                                                        self.batch_size)

            levels = self.get_levels(chunk)
            levels_vectors = self.vectorise_lists_dict(levels, self.batch_size)

            inputs = self.compile_inputs(feature_vectors)
            targets = self.compile_outputs(levels_vectors,
                                           self.params['output_classes'],
                                           self.output_map)

            return (inputs, targets)

            chunk_count += 1
            if chunk_count >= self.n_chunks:
                break
