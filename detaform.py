import pandas as pd
import re
import string
# df = pd.read_csv("datasets/full_text.txt",
#                  delimiter="\t",
#                  header=None,
#                  encoding='latin-1')
# df.columns = ["user", "time", "x", "lat", "lon", "text"]
# print(len(df['user'].unique()))
# print(df.info())
# tdf = df.sample(n=300000, random_state=42)
# print(len(tdf['user'].unique()))
# df = df.drop(df.sample(n=300000, random_state=42).index, axis=0)
# edf = df.sample(n=76000, random_state=42)
# print(df.info())
# print(len(edf['user'].unique()))
# df['place'] = ''
# df.drop('x', axis=1, inplace=True)
#
# print(len(df["user"].unique()))
# df_by_user = pd.DataFrame(columns=["lon", "lat", "text", "user"])
# df_by_user["user"] = df["user"].unique()
# print(df_by_user.info())
# print(df_by_user)
# for i in range(len(df_by_user.index)):
#     user = df_by_user.loc[df_by_user.index[i], "user"]
#     user_texts = df.loc[df['user'] == user]
#     df_by_user.loc[df_by_user.index[i], "text"] = '. '.join(user_texts["text"].astype(str))
#     df_by_user.loc[df_by_user.index[i], "lon"] = user_texts.loc[user_texts.index[0], "lon"]
#     df_by_user.loc[df_by_user.index[i], "lat"] = user_texts.loc[user_texts.index[0], "lat"]
#
# print(df)
#
# with open("datasets/eisenstein.jsonl", "w") as f:
#     df.to_json(f, orient='records', lines=True)

# def chunks(lst, n):
#     """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]
#
# df = pd.read_json(path_or_buf="datasets/eisenstein_user.jsonl", lines=True)
# print(df.info())
# print(df["text"])
# lengths = df["text"].str.len()
# print(lengths)
# count = df['text'].str.split().str.len()
# print(count)
#
# df_split = pd.DataFrame(columns=["lon", "lat", "text", "user"])
# for i in range(len(df.index)):
#     texts = df.loc[df.index[i], "text"]
#     words = len(texts.split())
#     tw = texts.split()
#     if words // 300 > 0:
#         n = words//300 + 1
#         size = words//n + 1
#         splitted = [tw[i:i + size] for i in range(0, len(tw), size)]
#         for k in range(len(splitted)):
#             df_short = df.iloc[i]
#             splitted[k] = " ".join(splitted[k])
#             df_short["text"] = splitted[k]
#             df_split = df_split.append(df_short, ignore_index=True)
#     else:
#         df_split = df_split.append(df.iloc[i], ignore_index=True)
#
#
# with open("datasets/eisenstein_user_test.jsonl", "w") as f:
#         df_split.to_json(f, orient='records', lines=True)

# def nlp_filtering(text):
#     def filter_punctuation(text):
#         punctuationfree="".join([i for i in text if i not in string.punctuation])
#         return punctuationfree
#
#     def filter_websites(text):
#         #pattern = r'(http\:\/\/|https\:\/\/)?([a-z0-9][a-z0-9\-]*\.)+[a-z][a-z\-]*'
#         pattern = r'http\S+'
#         text = re.sub(pattern, '', text)
#         return text
#
#     text = filter_websites(text)
#     text = filter_punctuation(text)
#     return text
#
# df = pd.read_json(path_or_buf="datasets/eisenstein_user_test.jsonl", lines=True)
# print(df.info())
# print(df["text"])
# lengths = df["text"].str.len()
# print(lengths)
# count = df['text'].str.split().str.len()
# print(count.max())
#
# df["text"] = df["text"].apply(nlp_filtering)
#
# count = df['text'].str.split().str.len()
# print(count.max())


df = pd.read_json(path_or_buf="results/val-data/U-NON-GEO+GEO-ONLY-O5-d-total_sum-mf_sum-pos_spher-weighted-N30e5-B10-E3-cosine-LR[1e-05;1e-06]_predicted_N100000_2022-10-29.jsonl", lines=True)
df = df.drop(df.sample(n=99900, random_state=42).index, axis=0)
with open("results/val-data/pmop-test.jsonl", "w") as f:
    df.to_json(f, orient='records', lines=True)

# size = 300000
# vs = 300000
# df = pd.read_json(path_or_buf="datasets/worldwide-twitter-day.jsonl", lines=True)
# print(df.info())
# print(len(df['lang'].unique()))
# print(len(df['code'].unique()))
# print(len(df['user'].unique()))
# tdf = df.sample(n=size, random_state=42)
# print(len(tdf['lang'].unique()))
# print(len(tdf['code'].unique()))
# print(len(tdf['user'].unique()))
# df = df.drop(df.sample(n=size, random_state=42).index, axis=0)
# edf = df.sample(n=vs, random_state=42)
# print(df.info())
# print(len(edf['lang'].unique()))
# print(len(edf['code'].unique()))
# print(len(edf['user'].unique()))
