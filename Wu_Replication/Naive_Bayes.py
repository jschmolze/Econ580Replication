import numpy as np 
import pandas as pd 
from sklearn.naive_bayes import MultinomialNB
import re

dir_data="./" # specify the directory to data files 
dir_lasso="./" # where the outputs are saved 

### (1) Identify training/test samples 
posts=pd.read_csv(dir_data+"gendered_posts.csv") 
keys_X=pd.read_csv(dir_data+'keys_to_X.csv') # in the same order as rows in matrix x

# additional step to make sure the order is the consistent with the matrix "X" of word counts 
# (This step may be unnecessary if you have sorted posts by title_id and post_id)
keys_merged=pd.merge(keys_X,posts,on=['title_id','post_id'],how="left") 

# note: "non-duplicate" posts contain only female or only male classifiers
i_train=np.where(keys_merged['training']==1) # 75% of non-duplicate posts as training sample
i_test0=np.where(keys_merged['training']==0) # 25% of non-duplicate posts as test sample for selecting optimal probability threshold
i_test1=np.where(keys_merged['training'].isnull()) # duplicate posts that include both female and male classifiers; To be reclassified 

# an array of unambiguous gender in the training sample 
y_train=keys_merged.loc[i_train[0],'female'].to_numpy() 

### (2) Bring in word count matrix X
word_counts = np.load(dir_data+"X_word_count.npz", encoding='latin1', allow_pickle=True)
X = word_counts['X'].item()
X_train=X[i_train[0],:]
X_test0=X[i_test0[0],:]
X_test1=X[i_test1[0],:]


### (3) Select Predictors: most frequent 10K excluding gender classifiers & additional last names 
vocab10K=pd.read_csv(dir_data+"vocab10K.csv")
vocab10K['exclude'].sum() 
exclude_vocab=vocab10K.loc[vocab10K['exclude']==1,:]
i_exclude=exclude_vocab['index']-1 # indexing in Python starts from 0, while the indices for vocab are 1 to 10,000

i_columns=range(10000)
i_keep_columns=list(set(i_columns)-set(i_exclude)) 
np.savetxt(dir_lasso+"i_keep_columns.txt",i_keep_columns) # later this can be merged by estimated coefficients (in the same order as these indices) 

X_train=X_train[:,i_keep_columns] 
print(X_train.shape)              
X_test0=X_test0[:,i_keep_columns] 
print(X_test0.shape)              
X_test1=X_test1[:,i_keep_columns] 
print(X_test1.shape)              

################################################################################################################
                                            ### Naive Bayes Model ###
################################################################################################################

# Initialize the model
# alpha=1.0 is Laplace smoothing, which prevents zero probabilities for words not seen in training
model_nb = MultinomialNB(alpha=1.0)

# Fit the model
model_nb.fit(X_train, y_train)

# Naive Bayes doesn't have "coefficients" in the same way LASSO does, 
# but it has log-probabilities of features given a class.
# This represents the 'importance' of words for the Female (1) class.
feature_log_probs = model_nb.feature_log_prob_[1] 
np.savetxt(dir_lasso + "nb_feature_log_probs.txt", feature_log_probs)

# Predicted probability for a post being Female
# predict_proba returns [Pr(0), Pr(1)], we take [:, 1] for Pr(female=1)
ypred_train_nb = model_nb.predict_proba(X_train)[:, 1]
ypred_test0_nb = model_nb.predict_proba(X_test0)[:, 1]
ypred_test1_nb = model_nb.predict_proba(X_test1)[:, 1]

# Save the predictions
np.savetxt(dir_lasso + "ypred_train_nb.txt", ypred_train_nb)
np.savetxt(dir_lasso + "ypred_test0_nb.txt", ypred_test0_nb)
np.savetxt(dir_lasso + "ypred_test1_nb.txt", ypred_test1_nb)

print("Naive Bayes training complete.")
print(f"Train set predictions mean: {ypred_train_nb.mean():.4f}")

vocab10K = pd.read_csv(dir_data + "vocab10K.csv")

# Filter to only keep the words we didn't exclude
vocab_filtered = vocab10K.iloc[i_keep_columns].copy()
feature_names = vocab_filtered['word'].values

# class 0 = Male, class 1 = Female 
log_prob_male = model_nb.feature_log_prob_[0]
log_prob_female = model_nb.feature_log_prob_[1]

# Calculate the Log-Odds Ratio
log_odds_ratio = log_prob_female - log_prob_male

# We use a mask to ignore symbols, numbers, and very short words
import re
def is_clean(w):
    # Returns True only if the word is strictly alphabetic and 3+ letters long
    return bool(re.match(r'^[a-z]{3,}$', str(w).lower()))

clean_mask = np.array([is_clean(w) for w in feature_names])

# Apply the filter to both the names and the scores
clean_feature_names = feature_names[clean_mask]
clean_log_odds = log_odds_ratio[clean_mask]

# Now find the indices based on the cleaned data
# Largest positive = Female-leaning | Largest negative = Male-leaning
top10_fem_idx = np.argsort(clean_log_odds)[-10:][::-1]
top10_male_idx = np.argsort(clean_log_odds)[:10]

# Create the final table using the cleaned arrays
distinctive_words_df = pd.DataFrame({
    'Rank': range(1, 11),
    'Top Male-Leaning Words': clean_feature_names[top10_male_idx],
    'Male-Leaning Score': clean_log_odds[top10_male_idx],
    'Top Female-Leaning Words': clean_feature_names[top10_fem_idx],
    'Female-Leaning Score': clean_log_odds[top10_fem_idx]
})

print("\n--- Top 10 Clean Distinctive Words (Log-Odds Ratio) ---")
print(distinctive_words_df.to_string(index=False))