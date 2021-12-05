import datasets
import pandas as pd
import nltk

import fasttext
import os
from collections import Counter
from multiprocessing import Pool
#!python -m spacy download en_core_web_sm
import en_core_web_sm
import logging
import pathlib
from spacy import displacy
from argparse import ArgumentParser

DOWNLOADS_DIR = os.path.expanduser('~/Downloads')
# curl https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -o ~/Downloads/lid.176.bin
PRETRAINED_MODEL_PATH = f'{DOWNLOADS_DIR}/lid.176.bin'

FORBIDDEN_TAGS = set(['NOUN', 'PRON', 'DET'])
fasttext_model = fasttext.load_model(PRETRAINED_MODEL_PATH)


NUMBER_PROCESSORS = 30


class DiscourseMarkersExtractor():
    def __init__(self, config, dataset):

        self.config = config
        self.dataset = dataset
        self.nlp = en_core_web_sm.load()

        self.bracket_mappings = {o: c for o, c in zip('{([', '})]')}
        self.open_brackets = set(self.bracket_mappings.keys())
        self.closed_brackets = set(self.bracket_mappings.values())

    def covert_dms_to_pattern(self, s):
        s = s.replace('"', '')
        doc = self.nlp(s)
        newString = s
        original_chars = s.lower()
        pattern_to_present = s.lower()
        starts_with_ner = False
        # reversed to not modify the offsets of other entities when substituting
        for e in reversed(doc.ents):
            start = e.start_char
            if start == 0:
                starts_with_ner = True
            end = start + len(e.text)

            newString = newString[:start] + e.label_ + newString[end:]
            original_chars = original_chars[:start] + original_chars[end:]
            pattern_to_present = pattern_to_present[:start] + \
                e.label_ + pattern_to_present[end:]

        if len(doc) > 0:
            if (str(doc[0].pos_)) not in FORBIDDEN_TAGS and original_chars.strip()[0:1].isalpha() and 'cardinal' not in newString.lower():
                return (s, newString.lower(), pattern_to_present)
        return s, "", ""  # looks like garbage

    def get_potential_marker_and_sentence(self, sent):
        if ',' in sent:
            marker, remainer = sent.split(',', 1)
            l = marker.split()
            if len(l) > 3:
                return (None, sent)
            return (' '.join(l), remainer)
        return (None, sent)

    def get_sentences_with_dms(self, sentences):
        if len(sentences) < 2:
            return None

        _, prev_sent = self.get_potential_marker_and_sentence(sentences[0])
        for sent in sentences[1:]:
            marker, curr_sent = self.get_potential_marker_and_sentence(sent)
            if marker:
                yield prev_sent, curr_sent, marker
            prev_sent = curr_sent

    def is_balanced_brackets(self, s):
        stack = []
        for c in s:
            if c in self.open_brackets:
                stack.append(c)  # push
            elif c in self.closed_brackets:
                if len(stack) == 0:
                    return False
                if self.bracket_mappings[stack.pop()] != c:
                    return False
        return len(stack) == 0

    def is_legimate_sentence(self, sent):
        if self.is_balanced_brackets(sent):
            lang, confidence = fasttext_model.predict(
                [sent.replace('\n', ' ')])
            if (lang[0][0] == self.config.expected_language and confidence[0][0] > self.config.language_confidence):
                if (self.config.low_threshold_of_tokens_in_sentence < len(nltk.word_tokenize(sent)) <
                        self.config.high_threshold_of_tokens_in_sentence):
                    return True
        return False

    def get_filename(self, index):
        return f'{self.config.output_dir}/chunk_{index:05d}.csv.gz'

    def build_dataframe_with_potential_discourse_markers(self, args):
        index, (start, end) = args
        result = []
        for text in self.dataset[start: end]['text']:
            sentences = nltk.tokenize.sent_tokenize(text)
            if sentences:
                tuples = self.get_sentences_with_dms(sentences)
                good_sentences = [t for t in tuples if all(
                    [self.is_legimate_sentence(s) for s in t[0:2]])]
                result.extend(good_sentences)
        dt = pd.DataFrame(result, columns=['sentence1', 'sentence2', 'marker'])
        dt.to_csv(self.get_filename(index), index=None, compression='gzip')
        return set(dt.marker)

    def sample_dataframe(self, dt, sz):
        tmp = dt.drop_duplicates(['sentence2'])
        if len(tmp) < sz:
            return tmp
        return tmp.sample(sz)

    def sample_file(self, index):
        dt = pd.read_csv(self.get_filename(index))
        dt = dt[dt.pattern.isin(self.relevant_dms)]
        return dt.groupby('pattern', as_index=False).apply(
            self.sample_dataframe, (self.config.sentences_per_pattern_sample//len(self.chunks)+1) * self.config.safe_factor)

    def add_pattern(self, index):
        filename = self.get_filename(index)
        dt = pd.read_csv(filename)
        dt['pattern'] = [self.from_dm_to_pattern.get(m, "") for m in dt.marker]
        dt = dt[dt.pattern != ""]
        dt.to_csv(self.get_filename(index), index=None, compression='gzip')
        return Counter(dt.pattern)

    def extract_dms(self):
        chunk_size = self.config.input_chunk_size
        self.chunks = [(i//chunk_size, (i, i+chunk_size))
                       for i in range(0, len(self.dataset), chunk_size)]
        logging.info("Dataset splitted")

        with Pool(self.config.number_of_processors) as pl:
            ll = pl.map(
                self.build_dataframe_with_potential_discourse_markers, self.chunks)
        logging.info("Potential discourse markers recognized")
        potential_dms = set.union(*ll, set())

        with Pool(self.config.number_of_processors) as pl:
            patterns = pl.map(self.covert_dms_to_pattern, potential_dms)
        logging.info("Potential discourse markers converted to patterns")
        self.from_dm_to_pattern = {m: p for m, p, _ in patterns if len(p) > 0}
        from_pattern_to_presentation = {
            p: presentation for _, p, presentation in patterns if len(p) > 0}

        with Pool(self.config.number_of_processors) as pl:
            patterns_counts = pd.DataFrame(sum(pl.map(self.add_pattern, range(len(self.chunks))), Counter()).most_common(),
                                           columns=['pattern', 'count'])
        patterns_counts['presentation'] = [
            from_pattern_to_presentation[p] for p in patterns_counts.pattern]
        patterns_counts.to_csv(
            f'{self.config.output_dir}/pattern_count.csv.gz', index=None, compression='gzip')
        logging.info("Added pattern to rows in csv files")
        self.relevant_dms = set(
            patterns_counts[0:self.config.number_of_patterns_to_sample]['pattern'])

        with Pool(self.config.number_of_processors) as pl:
            res_df = pd.concat(pl.map(self.sample_file, range(len(self.chunks)))).groupby('pattern', as_index=False).apply(
                self.sample_dataframe, self.config.sentences_per_pattern_sample).reset_index(drop=True)
        chunk_size = self.config.output_chunk_size
        for i, index in enumerate(range(0, len(res_df), chunk_size)):
            res_df[index:index+chunk_size].to_csv(f'{self.config.output_dir}/sample_{i:03d}.csv.gz',
                                                  index=None, compression='gzip')


def get_argparser():
    parser = ArgumentParser(
        description='Find positive and negative discourse markers')
    parser.add_argument('--number_of_processors',  default=30,
                        help='Number of CPU cores to be used')
    parser.add_argument('--sentences_per_pattern_sample', default=1000)
    parser.add_argument('--number_of_patterns_to_sample', default=1000)
    parser.add_argument('--input_chunk_size', default=10000,
                        help="size of chunk to be processed in parralel")
    parser.add_argument('--output_chunk_size', default=100000,
                        help="number of rows in chunk to be stored in the filesytem in parralel")
    parser.add_argument('--output_dir', default='/tmp/ilyashn')
    parser.add_argument('--safe_factor', default=20,
                        help="Sampling factor, used to find enough example if patterns are not evenly distributed beween files")
    parser.add_argument('--expected_language', default='__label__en')
    parser.add_argument('--language_confidence', default=0.75)
    parser.add_argument('--low_threshold_of_tokens_in_sentence', default=3)
    parser.add_argument('--high_threshold_of_tokens_in_sentence', default=32)
    return parser


def main():
    parser = get_argparser()
    configuration = parser.parse_args()
    logging.info("Started")
    wiki = pd.read_csv('~/Downloads/wiki.sample.csv.gz')[0:2010]
    #wiki = datasets.load_dataset("wikipedia", "2200501.en", split='train')
    logging.info("Dataset loaded")

    dmsExtractor = DiscourseMarkersExtractor(configuration, wiki)
    pathlib.Path(configuration.output_dir).mkdir(parents=True, exist_ok=True)
    dmsExtractor.extract_dms()

    logging.info("Finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
