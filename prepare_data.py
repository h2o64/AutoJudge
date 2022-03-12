#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1 - read in  the data.
df_train = pd.read_pickle(os.path.join('data', 'train.pkl'))
df_test = pd.read_pickle(os.path.join('data', 'test.pkl'))

# 2 - Perform any data cleaning and split into private train/test subsets,
# No data cleaning is required

# 3 - Split public train/test subsets. In this case the private training
# data will be used as the public data
df_public = df_train

df_public_train, df_public_test = train_test_split(
    df_public, test_size=0.2, random_state=57)
    # specify the random_state to ensure reproducibility

df_public_train.to_pickle(os.path.join('data', 'public', 'train.pkl'))
df_public_test.to_pickle(os.path.join('data', 'public', 'test.pkl'))

