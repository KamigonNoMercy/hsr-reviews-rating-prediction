# 🚂 Honkai: Star Rail Reviews - Rating Prediction (NLP Multiclass Classification)

This repository contains my **NLP multiclass classification project** trained on **1,000 English reviews** scraped from [Honkai: Star Rail on Google Play Store](https://play.google.com/store/apps/details?id=com.HoYoverse.hkrpgoversea) using `google-play-scraper`.

The goal is to predict a user's **star rating (1-5)** purely from the text content of their review.

👉 **Note:** The review data belongs to its original authors (Google Play users). This repo is **for educational purposes only**.
👉 The **scraping pipeline, preprocessing, EDA, and model experiments** are my own work.

---

## ✨ Features

- End-to-end NLP pipeline: **scraping -> EDA -> preprocessing -> text representation -> modeling -> imbalance handling**
- Two text representations compared:
  - **TF-IDF** (unigram + bigram, importance-based sparse vectors)
  - **Word2Vec** (locally trained 100-dim embeddings, averaged per review)
- Two classical ML models:
  - **Logistic Regression**
  - **Random Forest** (and Linear SVM in the ROS variant)
- **Hyperparameter tuning** with `GridSearchCV` (scoring = macro F1)
- **Imbalance handling** using:
  - `SMOTETomek` (SMOTE + Tomek Links hybrid)
  - `RandomOverSampler` (ROS)
- Domain-specific preprocessing (HSR slang normalization: `f2p -> free_to_play`, `e2 -> eidolon_2`, etc.)
- Negation-aware stopword removal (keeps `not`, `no`, `nor`, `never` to preserve sentiment)

---

## ⚙️ Tech Stack

- **Language:** Python 3
- **Scraping:** `google-play-scraper`, `langdetect`
- **NLP / Preprocessing:** `NLTK` (tokenizer, POS tagger, WordNet lemmatizer)
- **Text Representation:** `scikit-learn` (TF-IDF), `gensim` (Word2Vec)
- **Modeling:** `scikit-learn` (LogReg, RandomForest, LinearSVC, GridSearchCV)
- **Imbalance Handling:** `imbalanced-learn` (SMOTETomek, RandomOverSampler)
- **EDA / Viz:** `pandas`, `matplotlib`, `wordcloud`

---

## 📊 Dataset

- **Source:** Google Play Store - `com.HoYoverse.hkrpgoversea` (global server)
- **Size:** 1,000 English reviews (filtered from 1,400 raw multilingual reviews via `langdetect`)
- **Time window:** 5 October 2025 - 7 November 2025
- **Columns used:** `review_text`, `rating`, `review_date`
- **Rating distribution (highly imbalanced):**

| Rating | Count | % |
|:------:|:-----:|:---:|
| 1 ⭐    | 533  | 53.3% |
| 2 ⭐    | 81   | 8.1%  |
| 3 ⭐    | 60   | 6.0%  |
| 4 ⭐    | 61   | 6.1%  |
| 5 ⭐    | 265  | 26.5% |

This mirrors the real-world **"J-shaped" distribution** typical of app store reviews, users tend to leave feedback when they're either very angry or very happy, rarely in the middle.

---

## 🔍 Key Insights from EDA

1. **Rating 1 dominates at 53%.** At the time of scraping, the game's public Play Store rating was **3.9 ⭐ from 467K reviews**, but the **recent** reviews I scraped skew heavily negative, a sign of a community flashpoint happening during the Oct-Nov 2025 window.

2. **The "powercreep + Cyrene + greedy" theme.** Top dominant words in 1-2 star reviews are `powercreep`, `greedy`, `cyrene`, `kit`, `money`, `gacha`. Players were upset about a new character (Cyrene) being perceived as intentionally over-tuned to force spending, while older characters became irrelevant.

3. **5-star reviews are mostly short.** Positive reviews cluster at `good`, `fun`, `love`, `graphic`, `story`, `amazing`, typically 1-2 sentences. Negative reviews are **much longer and more detailed** (right-skewed word-count distribution), which is a common pattern in review data: unhappy users write essays, happy users write emojis.

4. **Middle ratings (2-4) are semantically mixed.** 3-star reviews contain both `good/fun/love` AND `powercreep/problem/issue`. This makes them genuinely hard to classify, which later shows up clearly in the model results.

5. **Domain-specific slang matters a lot.** Raw reviews are full of HSR-specific terms: `f2p`, `p2w`, `e0/e1/e2/e6`, `hp/atk/def/spd`, `MoC`, `AS`. Without normalization, the model treats `f2p` and `free to play` as two separate tokens, hurting generalization. A custom normalization dictionary fixed this.

---

## 🧪 Modeling Results

All numbers are on the held-out test set (200 samples, stratified). **Macro F1** is the fair metric here because of the severe class imbalance.

### Baseline (no imbalance handling)

| Model | Representation | Accuracy | Macro F1 |
|---|---|:---:|:---:|
| LogReg | TF-IDF | 0.685 | 0.294 |
| RandomForest | TF-IDF | 0.630 | 0.272 |
| LogReg | Word2Vec | 0.630 | 0.264 |
| RandomForest | Word2Vec | 0.650 | 0.275 |

👉 High accuracy is misleading, the models basically **only learn to predict rating 1 and 5**, completely ignoring ratings 2-4 (F1 = 0.00 on those classes).

### After Hyperparameter Tuning (GridSearchCV, scoring = macro F1)

| Model | Representation | Accuracy | Macro F1 |
|---|---|:---:|:---:|
| **LogReg (balanced)** | **TF-IDF** | 0.510 | **0.367** |
| RandomForest | TF-IDF | 0.660 | 0.284 |
| LogReg (balanced) | Word2Vec | 0.470 | 0.347 |
| RandomForest | Word2Vec | 0.655 | 0.277 |

👉 Using `class_weight="balanced"` **trades accuracy for fairness**, macro F1 jumps from 0.29 -> 0.37 because the model finally tries to predict minority classes.

### After SMOTETomek Oversampling

| Model | Representation | Accuracy | Macro F1 |
|---|---|:---:|:---:|
| **LogReg** | **TF-IDF** | 0.555 | **0.373** |
| RandomForest | TF-IDF | 0.585 | 0.265 |
| LogReg | Word2Vec | 0.450 | 0.331 |
| **RandomForest** | **Word2Vec** | 0.610 | **0.358** |

👉 **TF-IDF + Logistic Regression + SMOTETomek** ended up being the most balanced configuration. Macro F1 around **0.37** is modest in absolute terms, but given 1,000 samples, 5 imbalanced classes, and lots of semantic overlap in the middle ratings, it's a realistic ceiling for classical ML here.

---

## 🧠 Lessons Learned

- **Accuracy is a trap on imbalanced data.** A 68% accuracy model that predicts ratings 2-4 with F1 = 0.00 is worse than a 55% accuracy model that at least tries.
- **TF-IDF beat Word2Vec here.** With only ~800 training reviews, there isn't enough text to train a strong Word2Vec. Pretrained embeddings (GloVe, fastText, BERT) would almost certainly do better.
- **Domain normalization > fancier models.** Consolidating HSR slang gave a bigger boost than switching algorithms.
- **5-class rating prediction from short text is genuinely hard.** A 2-star and a 3-star review often say nearly identical things. Collapsing labels into `negative / neutral / positive` is a reasonable next step.

---

## 🔮 Future Work

1. **Simplify the label** from 5 classes to 3 (negative / neutral / positive) to reduce overlap between adjacent ratings.
2. **Use pretrained embeddings** (GloVe, fastText) or **fine-tune a transformer** (BERT, DistilBERT, IndoBERT for multilingual extension).
3. **Collect more data for minority classes** (ratings 2-4) through targeted scraping across longer time windows.
4. **Try cost-sensitive learning** beyond just `class_weight="balanced"`, e.g. focal loss with a neural model.
5. **Aspect-based sentiment analysis**, split each review into aspects (gameplay, monetization, story, performance) since HSR reviews usually complain about specific things, not the game as a whole.

---

## 🗂️ Repository Structure

```
hsr-reviews-rating-prediction/
├─ README.md
├─ HonkaiStarRailScrapping.ipynb              # Step 1: scraping + language filter
├─ EDA___Preprocessing_TextRepresentation_Model.ipynb   # Full end-to-end notebook
├─ TextRepresentation_Model.ipynb             # Standalone modeling notebook (TF-IDF, Word2Vec, SMOTETomek)
├─ data/
│  ├─ hsr_reviews_en.csv                      # 1,000 scraped English reviews
│  ├─ hsr_reviews_clean_preprocessed.csv      # After preprocessing (intermediate)
│  └─ hsr_reviews_clean_preprocessed_final.csv  # Final version used for modeling
└─ requirements.txt
```

---

## 📒 Notebooks Overview

| Notebook | What's inside |
|---|---|
| `HonkaiStarRailScrapping.ipynb` | Scrapes 1,400 raw reviews, filters to 1,000 English-only reviews using `langdetect` |
| `EDA___Preprocessing_TextRepresentation_Model.ipynb` | Full pipeline: EDA (rating distribution, wordclouds, top words per rating, review length), preprocessing (lowercasing, URL/HTML removal, slang normalization, negation-aware stopwords, POS-aware lemmatization), TF-IDF + Word2Vec, baseline + tuned + ROS models |
| `TextRepresentation_Model.ipynb` | Focused modeling notebook with TF-IDF, Word2Vec, GridSearchCV, and SMOTETomek |

---

## 🚀 How to Reproduce

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/hsr-reviews-rating-prediction.git
cd hsr-reviews-rating-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK resources (first run only)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# 4. (Optional) Re-scrape fresh reviews
jupyter notebook HonkaiStarRailScrapping.ipynb

# 5. Run the main pipeline
jupyter notebook EDA___Preprocessing_TextRepresentation_Model.ipynb
```

---

## License

- Code in this repository is released under the **MIT License**.
- The review data is scraped from Google Play Store and belongs to its respective users / Google. This repository does **not claim ownership** of the review content, it is included only for **non-commercial educational / portfolio purposes**. If you are a rights holder and want content removed, please open an issue.
- **Honkai: Star Rail** is a trademark of **HoYoverse / miHoYo**. This project is **not affiliated with, endorsed by, or sponsored by HoYoverse**.
