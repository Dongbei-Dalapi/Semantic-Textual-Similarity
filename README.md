# Semantic-Textual-Similarity
An implementation for SemEval-2016 Task1.

## Task Definition
Given two sentences, participating systems are asked to return a continuous valued similarity score on a scale from 0 to 5, with 0 indicating that the semantics of the sentences are completely independent and 5 signifying semantic equivalence.

## Usage

```
cd {project_folder/src}
python ensemble.py
```

## Data
### Training Data
Task participants are allowed to use all of the data sets released during prior years (2012-2015) as training data.
### Testing data
There are five source of testind data: Headline, Plagirism, Postediting, Question to Question and Answer to Answer.

## NLP Fature
We used two nlp features to capture useful information.
### N-gram overlap  
We calculated the similarity from the character n-grams extracted from two sentences.
### BOW cosine similarity
Each sentence is represented as a Bag-of-Words (BOW) and each word is weighted by its IDF value. The cosine similarity between two sentences is then calculated as a feature. We got 1 feature for BOW.

### Manhattan LSTM
There is two identical LSTM network. LSTM is passed word vector representations of sentences and output a hidden state encoding semantic meaning of the sentences using manhattan distance. 

## Ensemble model
We use Random Forests (RF), Gradient Boosting (GB),XGBoost (XGB) for traditional features and the LSTM model. We average the scores from four models to achieve better performance.

## Result

| NLP Features   |  Headline    | Plagiarsim   | Postediting  | Ans - Ans    | Ques - Ques  | All          |
| -------------  | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Ngram Overlap  | 0.7519       | 0.3726       | 0.4819       | 0.7942       | 0.5949       | 0.6327       |
| BOW similarity | 0.7228       | 0.3408       | 0.3666       | 0.7335       | 0.5669       | 0.5635       |
| Overlap + BOW  | 0.7409       | 0.3628       | 0.4339       | 0.7928       | 0.5855       | 0.6112       |          |
| All features   | 0.6423       | 0.7593       | 0.8061       | 0.5040       | 0.3780       | 0.6315       |


## Possible improvement
more features and more model tuning will be added later.

## Reference
- J. Tian, Z. Zhou, M. Lan, and Y. Wu. Ecnu at semeval-2017 task 1: Leverage kernel-based traditional nlp
features and neural networks to build a universal model for multilingual and cross-lingual semantic textual
similarity. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017),
pages 191–197, 2017.
- Mueller, J., & Thyagarajan, A. (2016, March). Siamese recurrent architectures for learning sentence similarity. In Proceedings of the AAAI conference on artificial intelligence (Vol. 30, No. 1).
