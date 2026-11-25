# -*- coding: UTF-8 -*-

""" Convert MovieLens-1M into the Top-k recommendation format required by ReChorus.

The pipeline strictly follows `MovieLens-1M.ipynb` shipped with the framework:
	1. keep ratings >= 4 as positive (implicit) feedback;
	2. 5-core filtering on users and items;
	3. split by timestamp: 80% / 10% / 10% (by day);
	4. drop dev/test interactions whose user or item is unseen in training;
	5. re-index users and items from 1;
	6. sample 99 negative items for every dev/test instance.

Output: data/MovieLens_1M/{train,dev,test}.csv  (tab separated)
Usage:  python prepare_ml1m.py
"""

import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET = 'ml-1m'
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(CUR_DIR, 'raw')
ZIP_PATH = os.path.join(RAW_DIR, DATASET + '.zip')
RAW_PATH = os.path.join(RAW_DIR, DATASET)
NEG_ITEMS = 99
RANDOM_SEED = 0


def download_and_extract():
	os.makedirs(RAW_DIR, exist_ok=True)
	if not os.path.exists(ZIP_PATH):
		url = 'https://files.grouplens.org/datasets/movielens/{}.zip'.format(DATASET)
		print('Downloading', url)
		urllib.request.urlretrieve(url, ZIP_PATH)
	if not os.path.exists(os.path.join(RAW_PATH, 'ratings.dat')):
		with zipfile.ZipFile(ZIP_PATH) as f:
			f.extractall(RAW_DIR)


def read_interactions():
	rows = []
	with open(os.path.join(RAW_PATH, 'ratings.dat'), encoding='latin-1') as f:
		for line in f:
			uid, iid, rating, time = line.strip().split('::')
			if float(rating) >= 4:
				rows.append([uid, iid, int(time)])
	return pd.DataFrame(rows, columns=['original_user_id', 'original_item_id', 'time'])


def five_core(df):
	# iterative 5-core filtering, identical in spirit to the official notebook
	while True:
		u_cnt = df['original_user_id'].value_counts()
		i_cnt = df['original_item_id'].value_counts()
		keep_u = u_cnt[u_cnt >= 5].index
		keep_i = i_cnt[i_cnt >= 5].index
		new_df = df[df['original_user_id'].isin(keep_u) & df['original_item_id'].isin(keep_i)]
		if len(new_df) == len(df):
			break
		df = new_df
	return df


def generate_negative(df, all_items, clicked_item_set, seed, neg_num=NEG_ITEMS):
	np.random.seed(seed)
	neg_items = np.random.choice(all_items, (len(df), neg_num))
	for i, uid in enumerate(tqdm(df['user_id'].values)):
		clicked = clicked_item_set[uid]
		for j in range(neg_num):
			while neg_items[i][j] in clicked or neg_items[i][j] in set(neg_items[i][:j]):
				neg_items[i][j] = np.random.choice(all_items, 1)[0]
	return neg_items.tolist()


def main():
	download_and_extract()
	df = read_interactions()
	print('Positive interactions before 5-core:', len(df))
	df = five_core(df).copy()
	print('Positive interactions after 5-core:', len(df))

	day = (pd.to_datetime(df['time'], unit='s').dt.normalize())
	df['day'] = (day - day.min()).dt.days
	max_day = df['day'].max()

	train = df[df['day'] <= int(max_day * 0.8)].copy()
	val_test = df[df['day'] > int(max_day * 0.8)].copy()
	dev = val_test[val_test['day'] <= int(max_day * 0.9)].copy()
	test = val_test[val_test['day'] > int(max_day * 0.9)].copy()

	train_u = set(train['original_user_id'])
	train_i = set(train['original_item_id'])
	dev = dev[dev['original_user_id'].isin(train_u) & dev['original_item_id'].isin(train_i)].copy()
	test = test[test['original_user_id'].isin(train_u) & test['original_item_id'].isin(train_i)].copy()

	# continuous ids beginning from 1
	merged = pd.concat([train, dev, test], axis=0)
	user2newid = dict(zip(sorted(merged['original_user_id'].unique()), range(1, merged['original_user_id'].nunique() + 1)))
	item2newid = dict(zip(sorted(merged['original_item_id'].unique()), range(1, merged['original_item_id'].nunique() + 1)))
	for part in [train, dev, test]:
		part['user_id'] = part['original_user_id'].map(user2newid)
		part['item_id'] = part['original_item_id'].map(item2newid)

	all_df = pd.concat([train, dev, test], axis=0)

	clicked_item_set = {uid: set(g['item_id']) for uid, g in pd.concat([train, dev, test]).groupby('user_id')}
	all_items = np.sort(all_df['item_id'].unique())
	dev['neg_items'] = generate_negative(dev, all_items, clicked_item_set, RANDOM_SEED + 1)
	test['neg_items'] = generate_negative(test, all_items, clicked_item_set, RANDOM_SEED + 2)

	train[['user_id', 'item_id', 'time']].sort_values(['user_id', 'time']).to_csv(
		os.path.join(CUR_DIR, 'train.csv'), sep='\t', index=False)
	dev[['user_id', 'item_id', 'time', 'neg_items']].sort_values(['user_id', 'time']).to_csv(
		os.path.join(CUR_DIR, 'dev.csv'), sep='\t', index=False)
	test[['user_id', 'item_id', 'time', 'neg_items']].sort_values(['user_id', 'time']).to_csv(
		os.path.join(CUR_DIR, 'test.csv'), sep='\t', index=False)

	print('Users: {}, Items: {}, Train: {}, Dev: {}, Test: {}'.format(
		all_df['user_id'].nunique(), all_df['item_id'].nunique(), len(train), len(dev), len(test)))
	print('Interactions: {}'.format(len(all_df)))


if __name__ == '__main__':
	main()
