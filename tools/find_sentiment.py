import os  
from multiprocessing import Pool
import pandas as pd
from scipy.stats import hypergeom
import nltk

from argparse import ArgumentParser

import logging

nltk.download('stopwords')
nltk.download('words')
stop_words = set(nltk.corpus.stopwords.words('english'))
words = set(nltk.corpus.words.words())



def get_p_value_from_dict(d):
    return hypergeom.sf(**d)


def get_p_value_from_dataframe(config, dt, column_name):
    M = sum(dt['total'])
    N = sum(dt[column_name])
    l = [{'k': raw[column_name]-1, 'N':N, 'n':raw['total'], 'M':M}
         for marker, raw in dt.iterrows()]
    with Pool(config.number_of_processors) as pl:
        return pl.map(get_p_value_from_dict, l)


def get_lexicon_score(list_of_strings):
    sets_of_tokens = sets_of_tokens = [
        set(nltk.word_tokenize(s.lower())) for s in list_of_strings]
    sets_of_non_stop_words = [st - stop_words for st in sets_of_tokens]
    sets_of_legitimate_tokens = [st.intersection(
        words) for st in sets_of_non_stop_words]

    number_of_uniq_legimite_tokens = len(
        set().union(*sets_of_legitimate_tokens))
    number_of_legimite_tokens = sum(
        [len(st) for st in sets_of_legitimate_tokens])

    number_of_empty_sets = sum(
        [len(st) == 0 for st in sets_of_legitimate_tokens])

    return number_of_uniq_legimite_tokens / (number_of_legimite_tokens + number_of_empty_sets)


def get_pos_neg_tuple_from_bert_score(config, bert_score):
    if bert_score < config.neg_prediction_threshold:
        sign = -1
    elif bert_score > config.pos_prediction_threshold:
        sign = 1
    else:
        sign = 0
    return (sign > 0, sign < 0,  sign == 0, sign)


def get_counts_for_discourse_markers_from_dataframe(config, df, path_prefix):

    df[['pos', 'neg', 'neutral', 'sign']] = [
        get_pos_neg_tuple_from_bert_score(config, x) for x in df['predicted_score']]

    cnt = df[['pattern', 'pos', 'neg', 'sign',
              'neutral']].groupby('pattern').sum()
#    cnt = cnt.join(patterns_entropy) # TODO add entropy

    cnt['total'] = df[['pattern']].groupby('pattern').size()

    logging.info("Starting calculating lexicon scores")
    with Pool(config.number_of_processors) as pl:
        l = [list(dt['sentence2']) for pattern, dt in df.groupby('pattern')]
        cnt['lexicon_score'] = pl.map(get_lexicon_score, l)
    logging.info("Starting calculating pvalues ")
    for column_name in ['pos', 'neg', 'neutral']:
        cnt[f'{column_name}_pvalue'] = get_p_value_from_dataframe(config, cnt, column_name)
        if column_name == 'neutral':
            cnt['neutral_ratio'] = cnt['neutral'] / (cnt['total'])
        else:
            cnt[f'{column_name}_ratio'] = cnt[column_name] / (cnt['neg'] + cnt['pos'])

    cnt.to_csv(f'{path_prefix}_cnt.csv')
    df.to_csv(f'{path_prefix}_raw.csv')
    return cnt, df


def filter_in_top_percent(columns, dt, percent):

    thresholds = {c: dt.sort_values(c).iloc[int(
        len(dt) * (1.0-percent))][c] for c in columns}
    for c, t in thresholds.items():
        dt = dt.query(f'{c} > {t}')
    return dt


def print_pos_neg(config, df):
    l = []
#    l = ['entropy', # TODO add ENTROPY
    l = ['lexicon_score']
    
    positive = set(filter_in_top_percent(l, df, config.top_percent_of_lexicon_score).query(
                   f'pos_pvalue<{config.pos_pvalue_threshold} and pos_ratio>{config.pos_ratio_threshold} and neutral_pvalue>{config.neutral_pvalue_threshold}').
                   sort_values('pos_ratio', ascending=False).reset_index()['pattern'])

    negative = set(filter_in_top_percent(l, df, config.top_percent_of_lexicon_score).query(
                   f'neg_pvalue<{config.neg_pvalue_threshold} and neg_ratio>{config.neg_ratio_threshold} and neutral_pvalue>{config.neutral_pvalue_threshold}').
                   sort_values('neg_ratio', ascending=False).reset_index()['pattern'])
    print('\n\tpos:\n\t', sorted(list(positive)))
    print('\tneg:\n\t', sorted(list(negative)))

def analyze_predictions(config):
    logging.info("Starting reading files with predictions")
    l = os.listdir(f"{config.predictions_dir}")
    bert_df = pd.concat([pd.read_csv(f"{config.predictions_dir}/{f}")
                        for f in l if f.endswith(".gz")])
    
    cnt, raw = get_counts_for_discourse_markers_from_dataframe(config,
        bert_df, f'./bert')
    print_pos_neg(config, cnt)



def get_argparser():
    parser = ArgumentParser(description='Find positive and negative discourse markers')
    parser.add_argument('--number_of_processors',  default=30, help='Number of CPU cores to be used')
    parser.add_argument('--predictions_dir', default='/Users/ilyashnayderman/Downloads/predictions', help='directory that contains gzipped files with scores')
    parser.add_argument('--pos_pvalue_threshold', default=0.00001)
    parser.add_argument('--pos_ratio_threshold', default=0.85)
    parser.add_argument('--neg_pvalue_threshold', default=0.00001)
    parser.add_argument('--neg_ratio_threshold', default=0.85)
    parser.add_argument('--neutral_pvalue_threshold', default=0.2)
    parser.add_argument('--top_percent_of_lexicon_score', default=0.67)
    parser.add_argument('--neg_prediction_threshold', default=0.1)
    parser.add_argument('--pos_prediction_threshold', default=0.9)


    return parser 

if __name__ == "__main__":
    parser = get_argparser()
    configuration = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    analyze_predictions(configuration)
