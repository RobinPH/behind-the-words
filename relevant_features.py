import concurrent.futures
import math

import numpy as np
import textstat
from lexicalrichness import LexicalRichness
from sentence_transformers import SentenceTransformer, util
from spacy.lang.char_classes import (ALPHA, ALPHA_LOWER, ALPHA_UPPER,
                                     CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS)
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from textblob import TextBlob


def custom_tokenizer(nlp):
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )

    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                     suffix_search=nlp.tokenizer.suffix_search,
                     infix_finditer=infix_re.finditer,
                     token_match=nlp.tokenizer.token_match,
                     rules=nlp.Defaults.tokenizer_exceptions)



class RelevantFeatures():
    def __init__(self, nlp, spellchecker, language_tool):
        self.nlp = nlp
        self.nlp.tokenizer = custom_tokenizer(self.nlp)
        self.spell = spellchecker
        self.tool = language_tool
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.__text_blob_cache = {}
        self.__text_lexical_richness_cache = {}
        self.__text_nlp_cache = {}
        self.__text_nlp_retokenized_cache = {}

        self.blacklist = ["_"]

        self.order = list(enumerate((self.flatten_features(self.get_features_raw(""))).keys()))

    def get(self, txt):
        features = self.get_features_raw(txt)

        return self.from_json(features)
    
    def from_json(self, _json):
      features = {}
      self.flatten_features(_json, None, features)
      features = self.fix_order(features)
      word_count = features["_word_count"]
      features = self.remove_blacklist(features)

      # lexical_diversity = lexical_diversity[:1] + lexical_diversity[3:6] + lexical_diversity[7:]
      mispelled_words = features["mispelled_words"] * word_count

      mispelled_words = 0 if mispelled_words < 3 else mispelled_words

      features["mispelled_words"] = mispelled_words
      features["grammar_score"] = easeInExpo(features["grammar_score"])
      features["formality_score"] = easeInExpo(features["formality_score"])

      return np.array(list(features.values()), dtype="float32")
    
    def fix_order(self, features):
      fixed = {}

      for index, key in self.order:
        if key in features:
          fixed[key] = features[key]
        else:
          raise Exception(f"Key '{key}' not found")

      return fixed

    def get_feature_names(self):
      _json = self.get_as_json("")
      _json = self.remove_blacklist(_json)

      return list(_json.keys())
    
    def remove_blacklist(self, _json):
      result = {**_json}

      for key in _json:
        remove = False

        for bl in self.blacklist:
          if key.startswith(bl):
            remove = True
            break

        if remove:
          del result[key]

      return result

    def remove_blacklist_list(self, lst):
      result = []

      for entry in lst:
        remove = False

        for bl in self.blacklist:
          if entry.startswith(bl):
            remove = True
            break

        if not remove:
          result.append(entry)
          

      return result

    def get_as_json(self, txt):
        # mispelled_words = self.get_mispelled_words(txt)
        # word_count = mispelled_words["word_count"]
        # mispelled_words = 0 if mispelled_words["word_count"] == 0 else len(mispelled_words["mispelled_words"]) / mispelled_words["word_count"] ##

        # grammar_score = self.get_grammar_score(txt) ##

        # repetitive_words = self.get_repetitive_words(txt)
        # repetitive_words = np.array([cnt for _, cnt in repetitive_words])
        # ptp = 0 if len(repetitive_words) == 0 else np.ptp(repetitive_words)
        # repetitive_words = repetitive_words if ptp == 0 else (repetitive_words - np.min(repetitive_words)) / ptp ##
        # repetitive_words = resize_array(repetitive_words.tolist(), 128)

        # formality_score = self.get_formality_score(txt) ##

        # readability_score = self.get_readability_score(txt)
        # readability_score = [value / 100 for value in readability_score.values()] ##

        # metaphor_usage_score = self.get_metaphor_usage_score(txt) ##
        # metaphor_usage_score = metaphor_usage_score["score"]
        
        # lexical_diversity = self.get_lexical_diversity(txt) ##
        # lexical_diversity = list(lexical_diversity.values())

        # sentiment = self.get_sentiment(txt) ##
        # subjectivity = self.get_subjectivity(txt) ##

        return self.get_features(txt)
    
    def _get_features_raw(self, txt, concurrence=False):
      import time
      def foo(label, func):
        def bar(txt):
          start = time.time()
          # print("Started", label)
          res = func(txt)
          # print(f"Finished ({round(time.time() - start, 4)}s)", label)
          return res
        
        return bar
      

      functions = [
        foo("self.get_mispelled_words", self.get_mispelled_words),
        foo("self.get_grammar_score", self.get_grammar_score),
        foo("self.get_repetitive_words", self.get_repetitive_words),
        foo("self.get_coherence_score", self.get_coherence_score),
        foo("self.get_formality_score", self.get_formality_score),
        foo("self.get_readability_score", self.get_readability_score),
        foo("self.get_metaphor_usage_score", self.get_metaphor_usage_score),
        foo("self.get_lexical_diversity", self.get_lexical_diversity),
        foo("self.get_sentiment", self.get_sentiment),
        foo("self.get_subjectivity", self.get_subjectivity),
      ]

      if concurrence:
        print("concurrence")
        results = [None] * len(functions)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(functions)) as executor:
          executions = {executor.submit(func, txt): idx for idx, func in enumerate(functions)}
          for future in concurrent.futures.as_completed(executions):
              idx = executions[future]

              result = future.result()

              results[idx] = result

        return results
      else:
        return [func(txt) for func in functions]
    
    def get_features_raw(self, txt, features_raw = None):
        if features_raw == None:
          features_raw = self._get_features_raw(txt)

        features = {}

        for feature_raw in features_raw:
          for key in feature_raw["result"]:
            features[key] = feature_raw["result"][key]

        return features

    def get_features(self, txt):
        features_raw = self.get_features_raw(txt)

        features = {}
        self.flatten_features(features_raw, None, features)

        return features


    def flatten_features(self, value, key=None, result = {}):
      if key and type(value) != dict and type(value) != list:
        result[key] = value
      else:
        if type(value) == dict:
          for k in value:
            if key:
              _key = f"{key}.{k}"
            else:
              _key = k
            
            self.flatten_features(value[k], _key, result)
        elif type(value) == list:
          for idx, val in enumerate(value):
            if key:
              _key = f"{key}[{idx}]"
            else:
              _key = "[{idx}]"
            
            self.flatten_features(val, _key, result)
        else:
          print(key, value)
          raise Exception("Unhandled type of variable 'value'")

      return result
    
    def get_flattern_keys(self, value, key=None, keys=None):
      if keys == None:
        keys = []

      if key and type(value) != dict and type(value) != list:
        keys.append(key)
      else:
        if type(value) == dict:
          for k in value:
            if key:
              _key = f"{key}.{k}"
            else:
              _key = k
            
            self.get_flattern_keys(value[k], _key, keys)
        elif type(value) == list:
          for idx, val in enumerate(value):
            if key:
              _key = f"{key}[{idx}]"
            else:
              _key = "[{idx}]"
            
            self.get_flattern_keys(val, _key, keys)
        else:
          print(key, value)
          raise Exception("Unhandled type of variable 'value'")

      return keys

    #########################
    # Syntactic Measurement #
    #########################


    #  Get misspelt words
    ## https://pypi.org/project/spacy/
    ## https://pypi.org/project/pyenchant/
    ## https://stackoverflow.com/questions/59579049/how-to-tell-spacy-not-to-split-any-words-with-apostrophs-using-retokenizer
    def get_mispelled_words(self, txt):
      # is_mispelled_rule = lambda rule: rule.message == 'Possible spelling mistake found.' and len(rule.replacements) and not rule.replacements[0][0].isupper()

      # matches = self.tool.check(txt)

      # matches = [rule for rule in matches if is_mispelled_rule(rule)]

      # matches = [(txt[rule.offset:(rule.offset + rule.errorLength)], rule.replacements[0]) for rule in matches]

      # return matches

      doc = self.get_nlp_retokenized(txt)

      mispelled_words = []

      for token in doc:
        if not token.is_punct and not token.text[0].isupper() and token.text[0] != ' ':
          correction = self.spell.correction(token.text)

          if correction and correction != token.text:
            mispelled_words.append({ "text": token.text, "suggestion": correction, "offset": token.idx })

      # return {
      #   "mispelled_words": mispelled_words,
      #   "word_count": len(doc)
      # }
    
      return {
        "id": "mispelled_words",
        "metadata": {
          "mispelled_words": mispelled_words,
        },
        "result": {
          "mispelled_words": 0 if len(doc) == 0 else len(mispelled_words) / len(doc),
          "_word_count": len(doc),
        }
      }


    #  Get grammar error score
    ## https://pypi.org/project/language-tool-python/
    def get_grammar_score(self, txt):
      is_bad_rule = lambda rule: rule.message == 'Possible spelling mistake found.' and len(rule.replacements) and rule.replacements[0][0].isupper()
      errors = self.tool.check(txt)
      errors = [rule for rule in errors if not is_bad_rule(rule)]


      errors_formatted = [{ "message": error.message, "context": error.context, "offsetInContext": error.offsetInContext, "errorLength": error.errorLength } for error in errors]

      words = self.get_text_blob(txt).words

      if len(words) == 0:
        grammar_score = 0
      else:
        grammar_score = max(0, min(1, 1 - (len(errors) / len(words))))

      return {
        "id": "grammar_score",
        "metadata": {
          "errors_formatted": errors_formatted,
        },
        "result": {
          "grammar_score": grammar_score,
        }
      }


    # Get repetitive words score
    ## https://textblob.readthedocs.io/en/dev/quickstart.html#tokenization
    def get_repetitive_words(self, txt):
      doc = self.get_nlp_retokenized(txt)

      freq = {}

      for token in doc:
        word = token.text.lower()

        if word not in freq:
          freq[word] = 0

        freq[word] += 1

      filtered = filter(lambda item: item[1] > 1, freq.items())
      repetitive_words_raw = sorted(filtered, key=lambda item: -item[1])
      # return filtered

      # repetitive_words = np.array([cnt for _, cnt in repetitive_words_raw])
      # ptp = 0 if len(repetitive_words) == 0 else np.ptp(repetitive_words)
      # repetitive_words = repetitive_words if ptp == 0 else (repetitive_words - np.min(repetitive_words)) / ptp ##
      # repetitive_words = resize_array(repetitive_words.tolist(), 128)

      if len(repetitive_words_raw) > 0:
        repetitive_words_counts = [cnt for _, cnt in repetitive_words_raw]
        repetitive_words = (sum(repetitive_words_counts) - len(repetitive_words_raw)) / len(doc)
      else:
        repetitive_words = 0


      return {
        "id": "repetitive_words",
        "metadata": {
          "repetitive_words": repetitive_words_raw,
          "_word_count": len(doc),
        },
        "result": {
          "repetitive_words": repetitive_words,
        }
      }
    

    ########################
    # Semantic Measurement #
    ########################

    # Get Coherence Score
    def get_coherence_score(self, txt):
      doc = self.get_nlp(txt)
      sentences = [sent.text for sent in doc.sents]

      embeddings = self.sentence_model.encode(sentences)

      sentence_scores = []
      averages = []

      for i in range(len(sentences)):
        scores = []

        for j in range(len(sentences)):
          score = util.cos_sim(embeddings[i], embeddings[j])[0][0]
          scores.append(score.item())

        average = sum(scores) / len(scores)

        averages.append(average)

        sentence_scores.append({
          "sentence": sentences[i],
          "scores": scores,
          "average": average,
        })

      if len(averages) > 0:
        coherence_score = sum(averages) / len(averages)
      else:
        coherence_score = 0
      
      return {
        "id": "coherence_score",
        "metadata": {
          "sentence_scores": sentence_scores,
        },
        "result": {
          "coherence_score": coherence_score
        }
      }


    # Formality Score
    ## http://pespmc1.vub.ac.be/Papers/Formality.pdf page 13
    ## F = (noun frequency + adjective freq. + preposition freq. + article freq. – pronoun freq. – verb freq. – adverb freq. – interjection freq. + 100)/2
    def get_formality_score(self, txt):
      doc = self.get_nlp(txt)

      def get_words(rule):
        words = []

        for token in doc:
          if rule(token.lower_, token.pos_):
            words.append(token.text)

        return words
      
      article_words = ["the", "a", "an"]

      nouns = get_words(lambda _, tag: tag == "NOUN")
      adjectives = get_words(lambda _, tag: tag == "ADJ")
      prepositions = get_words(lambda _, tag: tag == "ADP")
      articles = get_words(lambda word, _: word in article_words)
      pronouns = get_words(lambda _, tag: tag == "PRON")
      verbs = get_words(lambda _, tag: tag == "VERB")
      adverbs = get_words(lambda _, tag: tag == "ADV")
      interjections = get_words(lambda _, tag: tag == "INTJ")

      # return { "nouns": nouns, "adjectives": adjectives, "prepositions": prepositions, "articles": articles, "pronouns": pronouns, "verbs": verbs, "adverbs": adverbs, "interjections": interjections }
      
      n_formal = len(nouns + adjectives + prepositions + articles)
      n_contextual = len(pronouns + verbs + adverbs + interjections)
      N = n_formal + n_contextual

      if N == 0:
        formality_score = 0
      else:
        formality_score = 50 * ((n_formal - n_contextual) / N + 1) / 100

      return {
        "id": "formality_score",
        "metadata": {
          "nouns": nouns,
          "adjectives": adjectives,
          "prepositions": prepositions,
          "articles": articles,
          "pronouns": pronouns,
          "verbs": verbs,
          "adverbs": adverbs,
          "interjections": interjections,
          "n_formal": n_formal,
          "n_contextual": n_contextual,
        },
        "result": {
          "formality_score": formality_score
        }
      }

    # Readability score
    ## https://pypi.org/project/textstat/
    def get_readability_score(self, txt):
      return {
        "id": "readability_score",
        "result": {
          "readability_score": {
            "flesch_reading_ease": textstat.flesch_reading_ease(txt),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(txt),
            "smog_index": textstat.smog_index(txt),
            "coleman_liau_index": textstat.coleman_liau_index(txt),
            "automated_readability_index": textstat.automated_readability_index(txt),
            "dale_chall_readability_score": textstat.dale_chall_readability_score(txt),
            # "difficult_words": textstat.difficult_words(txt),
            "linsear_write_formula": textstat.linsear_write_formula(txt),
            "gunning_fog": textstat.gunning_fog(txt),
          }
        }
      }

    # Metaphor usage score
    ## https://arxiv.org/pdf/1808.09653.pdf
    def get_metaphor_usage_score(self, txt):
      doc = self.get_nlp(txt)

      sentences = []

      for sentence in doc.sents:
        sentence = str(sentence)
        labels = self.metaphor_usage.predict(sentence)

        score = sum(labels) / len(labels) if len(labels) != 0 else 0

        sentences.append({
          "sentence": sentence,
          "score": score
        })

      if len(sentences) > 0:
        score = sum([entry["score"] for entry in sentences]) / len(sentences)
      else:
        score = 0

      return {
        "id": "metaphor_usage_score",
        "metadata": {
          "sentences": sentences,
        },
        "result": {
          "metaphor_usage_score": score,
        }
      }


    # Lexical Diversity
    ## https://pypi.org/project/lexicalrichness/
    def get_lexical_diversity(self, txt):
      lexical_richness = self.get_lexical_richness(txt)

      if lexical_richness.words == 0:
        return {
          "id": "lexical_diversity",
          "result": {
            "lexical_diversity": {
              "ttr": 0,
              "rttr": 0,
              "cttr": 0,
              "mtld": 0,
              "herdan": 0,
              "summer": 0,
              "maas": 0,
              "yulek": 0,
              "herdanvm": 0,
              "simpsond": 0,
            }
          }
        }

      return {
        "id": "lexical_diversity",
        "result": {
          "lexical_diversity": {
            "ttr": self.try_or_default(lambda: lexical_richness.ttr, 0),
            "rttr": self.try_or_default(lambda: lexical_richness.rttr, 0),
            "cttr": self.try_or_default(lambda: lexical_richness.cttr, 0),
            "mtld": self.try_or_default(lambda: lexical_richness.mtld(threshold=0.72), 0),
            "herdan": self.try_or_default(lambda: lexical_richness.Herdan, 0),
            "summer": self.try_or_default(lambda: lexical_richness.Summer, 0),
            "maas": self.try_or_default(lambda: lexical_richness.Maas, 0),
            "yulek": self.try_or_default(lambda: lexical_richness.yulek, 0),
            "herdanvm": self.try_or_default(lambda: lexical_richness.herdanvm, 0),
            "simpsond": self.try_or_default(lambda: lexical_richness.simpsond, 0),
          }
        }
      }

    #########################
    # Sentiment Measurement #
    #########################


    # Get sentiment
    ## https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis
    def get_sentiment(self, txt):
      return {
        "id": "sentiment",
        "result": {
          "sentiment": self.get_text_blob(txt).sentiment.polarity
        }
      }


    # Subjectivity score
    ## https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis
    def get_subjectivity(self, txt):
      return {
        "id": "subjectivity",
        "result": {
          "subjectivity": self.get_text_blob(txt).sentiment.subjectivity
        }
      }


    ## Utility

    def set_metaphor_usage_instance(self, metaphor_usage_instance):
        self.metaphor_usage = metaphor_usage_instance

    def get_text_blob(self, txt):
      if txt not in self.__text_blob_cache:
        self.__text_blob_cache[txt] = TextBlob(txt)

      return self.__text_blob_cache[txt]


    def get_lexical_richness(self, txt):
      if txt not in self.__text_lexical_richness_cache:
        self.__text_lexical_richness_cache[txt] = LexicalRichness(txt)

      return self.__text_lexical_richness_cache[txt]
    
    def get_nlp_retokenized(self, txt):
      if txt not in self.__text_nlp_retokenized_cache:
        doc = self.get_nlp(txt, use_cache=False)

        position = [token.i for token in doc if token.i != 0 and "'" in token.text and len(token.text.replace("'", "")) >= 1]

        with doc.retokenize() as retokenizer:
            for pos in position:
              try:
                retokenizer.merge(doc[pos-1:pos+1])
              except:
                pass

        self.__text_nlp_retokenized_cache[txt] = doc

      return self.__text_nlp_retokenized_cache[txt]

    def get_nlp(self, txt, use_cache=True):
      if not use_cache:
        return self.nlp(txt)
      
      if txt not in self.__text_nlp_cache:
        self.__text_nlp_cache[txt] = self.nlp(txt)


      return self.__text_nlp_cache[txt]
    
    def clear_cache(self):
      self.__text_blob_cache = {}
      self.__text_lexical_richness_cache = {}
      self.__text_nlp_cache = {}
      self.__text_nlp_retokenized_cache = {}

    def try_or_default(self, func, default_value):
      try:
        val = func()

        if math.isnan(val):
          raise Exception("value is nan")

        return val
      except:
        return default_value


def resize_array(array, length, default_item=0):
  array = array[:length]
  pad_length = max(0, length - len(array))
  
  if pad_length > 0:
    array = array + [default_item] * pad_length

  return array

def easeInExpo(x):
  return 0 if x == 0 else pow(2, 10 * x - 10);

def easeInExpo2(x):
  return 0 if x <= 0.2 else pow(2, 10 * x - 10);

def sigmoid(x):
  return 1 / (1 + math.exp(-x))