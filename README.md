# Behind the Words

## Install Dependencies
Run `pip install -r requirements.txt`

## Pretrained Models
Models for CNN, XGBoost, and XGBoost + CNN have already been trained. \
**You can skip to [`Step 4 (Backend Server)`](#step-4-backend-server), unless you want to preprocess, and train on your own.**

## Preprocess, and Train
### Step 0 (Optional)
Following this step will cause the following steps to reuse the Word2vec model [`word2vec-google-news-300`](data/gensim/word2vec-google-news-300.gz) served via HTTP
1. Open a terminal
2. Run [`0-word2vec_server.py`](0-word2vec_server.py), it might take a minute to load [`data/gensim/word2vec-google-news-300.gz`](data/gensim/word2vec-google-news-300.gz)

### Step 1 (CNN)
1. Run All [`1.1-preprocess_cnn.ipynb`](1.1-preprocess_cnn.ipynb) \
   Preprocess the word embeddings of each data in [`data/dataset-v1/cnn/*`](data/dataset-v1/cnn), and outputs it in `data/processed-v1/cnn/word-embedding`.
2. Run All [`1.2-train_cnn.ipynb`](1.2-train_cnn.ipynb) \
   Trains the CNN model. Outputs it in [`models/cnn`](models/cnn)

### Step 2 (XGBoost Relevant Features)
1. Run All [`2.1-preprocess_rf_features.ipynb`](2.1-preprocess_rf_features.ipynb) \
   Preprocess the releveant features of each data in [`data/dataset-v1/rf/*`](data/dataset-v1/rf), and outputs it in `data/processed-v1/rf/features`.
2. Run All [`2.2-train_rf.ipynb`](2.2-train_rf.ipynb) \
   Trains the XGBoost Relevant Features model. Outputs it in [`models/rf`](models/rf)

### Step 3 (XGBoost Relevant Features + CNN)
1. Run All [`3.1-preprocess_rf+cnn_features.ipynb`](3.1-preprocess_rf+cnn_features.ipynb) \
   Preprocess the releveant features of each data in [`data/dataset-v1/rf-cnn/*`](data/dataset-v1/rf-cnn), and outputs it in `data/processed-v1/rf-cnn/features`.
2. Run All [`3.2-preprocess_rf+cnn_prediction.ipynb`](3.2-preprocess_rf+cnn_prediction.ipynb) \
   Preprocess the CNN prediction of each data in [`data/dataset-v1/rf-cnn/*`](data/dataset-v1/rf-cnn), and outputs it in `data/processed-v1/rf-cnn/cnn-prediction`.
3. Run All [`3.3-train_rf+cnn.ipynb`](3.3-train_rf+cnn.ipynb) \
   Trains the XGBoost Relevant Features + CNN model. Outputs it in [`models/rf-cnn`](models/rf-cnn)

### Step 4 (Backend Server)
1. Run All [`4-backend_server.ipynb`](4-backend_server.ipynb) \
   Starts the backend server, it uses the latest model in [`models/cnn`](models/cnn), [`models/rf`](models/rf), and [`models/rf-cnn`](models/rf-cnn)
