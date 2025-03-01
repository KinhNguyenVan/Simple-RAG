import nltk
import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np
import google.generativeai as genaipip
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity


class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])


class MyModel():
    def __init__(self, contents):
        self.contents = contents
        self.documents = []
        self.extractor = KeyphraseExtractionPipeline(model="ml6team/keyphrase-extraction-distilbert-inspec")
        self.model_encoder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        self.single_contents = []
        self.single_contents_BGE = []
        self.embeddings = None
        self.embeddings_BGE = None

    @staticmethod
    def text_processing(text, process_stopwords=True):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()

        word_tokens = word_tokenize(text)

        if process_stopwords:
            filtered_sentence = [w for w in word_tokens if w not in stop_words]
        else:
            filtered_sentence = word_tokens

        filtered_sentence = [lemmatizer.lemmatize(w) for w in filtered_sentence]

        return " ".join(filtered_sentence)

    def processor(self):
        self.documents = []

        for s in self.contents:
            sentences = [sentence.strip() for sentence in re.split(r'\. ', s) if sentence.strip()]
            self.documents.extend(sentences)

        for i in range(len(self.documents)):
            self.single_contents.append(self.text_processing(self.documents[i]))
            self.single_contents_BGE.append(self.text_processing(self.documents[i], process_stopwords=False))

        return self

    def get_embeddings(self):
        self.embeddings = self.model_encoder.encode(
            self.single_contents,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )
        self.embeddings_BGE = self.model_encoder.encode(
            self.single_contents_BGE,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )
        return self

    @staticmethod
    def tf(term, content):
        if len(content.split()) == 0:
            return 0
        return content.split().count(term) / len(content.split())

    @staticmethod
    def idf(term, contents):
        doc_count = sum(1 for content in contents if term in content.split())
        return np.log(len(contents) / (1 + doc_count))

    def get_scores1(self, query_embedded):
        colbert_scores = np.array([
            self.model_encoder.colbert_score(query_embedded['colbert_vecs'], self.embeddings_BGE['colbert_vecs'][i])
            for i in range(len(self.embeddings_BGE['colbert_vecs']))
        ])
        sparse_scores = np.array([
            self.model_encoder.compute_lexical_matching_score(query_embedded['lexical_weights'], self.embeddings_BGE['lexical_weights'][i])
            for i in range(len(self.embeddings_BGE['lexical_weights']))
        ])
        dense_scores = np.array([
            query_embedded['dense_vecs'].dot(self.embeddings_BGE['dense_vecs'][i])
            for i in range(len(self.embeddings_BGE['dense_vecs']))
        ])
        return colbert_scores + dense_scores + sparse_scores

    def get_scores2(self, query_embedded, query_keywords, query_processed):
        keyword_weight = np.zeros(len(self.single_contents))
        if query_keywords == "":
            query_keywords = query_processed
        for keyword in query_keywords.split():
            for i in range(len(self.single_contents)):
                similarity = cosine_similarity(
                    query_embedded, self.embeddings['dense_vecs'][i].reshape(1, -1)
                )[0, 0]
                other_similarity = [
                    cosine_similarity(
                        self.embeddings['dense_vecs'][j].reshape(1, -1),
                        self.embeddings['dense_vecs'][i].reshape(1, -1)
                    )[0, 0]
                    for j in range(len(self.single_contents)) if i != j
                ]
                if similarity > 0.1:
                    keyword_weight[i] += (
                        self.idf(keyword, self.single_contents)
                        * self.tf(keyword, self.single_contents[i])
                        * similarity
                        / (1 - np.mean(other_similarity) + 1e-6)
                    )
        keyword_weight = (keyword_weight - np.min(keyword_weight)) / (np.max(keyword_weight) - np.min(keyword_weight))
        return keyword_weight

    def fit(self):
        self.processor()
        self.get_embeddings()

    def make_response(self, query, alpha=0.1):
        query_BGE = self.text_processing(query, process_stopwords=False)
        query_embedding_BGE = self.model_encoder.encode(
            query_BGE,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )

        query_processed = self.text_processing(query)
        query_keywords = " ".join(self.extractor(query_processed))
        query_embedding = self.model_encoder.encode(
            query_processed,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )['dense_vecs'].reshape(1, -1)

        scores1 = self.get_scores1(query_embedding_BGE)
        scores2 = self.get_scores2(query_embedding, query_keywords, query_processed)
        scores = scores1 + scores2

        std_dev = np.std(scores)
        mean_value = np.mean(scores)
        thresholded_weights = [
            (i, kw) for i, kw in enumerate(scores) if kw > (mean_value + alpha * std_dev)
        ]
        top_k_indices = sorted(thresholded_weights, key=lambda x: x[1], reverse=True)
        top_k_indices = [i for i, _ in top_k_indices]
        list_ans = [self.documents[i] for i in top_k_indices]
        sentences = '. '.join(list_ans)

        # API
        genai.configure(api_key="")
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Prompt
        prompt = f"""
        This is a text containing information about a specific topic:

        {sentences}

        Based on this text, please answer the following question:

        {query}

        Please create a complete and clear answer, using knowledge only from the above text.
        """
        response = model.generate_content(prompt)
        result = response.text

        return result
