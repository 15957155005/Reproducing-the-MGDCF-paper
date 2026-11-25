# -*- coding: UTF-8 -*-

""" Small dev-based grid search for the MGDN encoder of Hetero-MGDCF.

The default setting of the paper (K=4, self-loops, alpha=0.1, beta=0.9) is tuned on
sparse graphs (Gowalla/Yelp/Amazon). MovieLens-1M is much denser, so the optimal
number of hops / self-loop option can differ; this script searches a few candidates
and writes ../results/grid_{dataset}.csv

	python scripts/run_grid.py --dataset MovieLens_1M --jobs 4 --epoch 80
"""

import os
import re
import csv
import sys
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
LOG_DIR = os.path.join(ROOT, 'log', 'grid')
RESULT_DIR = os.path.join(ROOT, 'results')

CANDIDATES = [
	(4, 1), (3, 1), (2, 1), (4, 0), (3, 0), (2, 0),
]
METRIC_RE = re.compile(r'Dev  After Training: .*NDCG@20:([\d.]+).*')
TEST_RE = re.compile(r'Test After Training: .*NDCG@20:([\d.]+)')


def log_path(dataset, k, self_loop):
	return os.path.join(LOG_DIR, 'MGDCF__{}__K{}_loop{}.txt'.format(dataset, k, self_loop))


def run_one(dataset, k, self_loop, epoch):
	cmd = [sys.executable, 'main.py', '--model_name', 'MGDCF', '--dataset', dataset,
		   '--path', '../data/', '--random_seed', '0', '--epoch', str(epoch),
		   '--log_file', log_path(dataset, k, self_loop),
		   '--emb_size', '64', '--n_layers', str(k), '--self_loop', str(self_loop),
		   '--alpha', '0.1', '--beta', '0.9', '--loss_type', 'InfoBPR', '--num_neg', '8',
		   '--lr', '1e-3', '--l2', '0', '--batch_size', '2048', '--num_workers', '0',
		   '--save_final_results', '0', '--topk', '20', '--metric', 'NDCG,HR',
		   '--main_metric', 'NDCG@20', '--early_stop', '10']
	env = dict(os.environ)
	env['OMP_NUM_THREADS'] = '6'
	env['MKL_NUM_THREADS'] = '6'
	print('[START] K={} self_loop={}'.format(k, self_loop), flush=True)
	with open(log_path(dataset, k, self_loop), 'w', encoding='utf-8') as f:
		subprocess.run(cmd, cwd=SRC, stdout=f, stderr=subprocess.STDOUT, env=env)
	print('[DONE ] K={} self_loop={}'.format(k, self_loop), flush=True)


def parse(dataset, k, self_loop):
	path = log_path(dataset, k, self_loop)
	if not os.path.exists(path):
		return None, None
	dev = test = None
	with open(path, encoding='utf-8', errors='ignore') as f:
		for line in f:
			m = METRIC_RE.search(line)
			if m:
				dev = float(m.group(1))
			m = TEST_RE.search(line)
			if m:
				test = float(m.group(1))
	return dev, test


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='MovieLens_1M')
	parser.add_argument('--epoch', type=int, default=80)
	parser.add_argument('--jobs', type=int, default=4)
	args = parser.parse_args()

	os.makedirs(LOG_DIR, exist_ok=True)
	os.makedirs(RESULT_DIR, exist_ok=True)
	with ThreadPoolExecutor(max_workers=args.jobs) as pool:
		futures = [pool.submit(run_one, args.dataset, k, s, args.epoch) for k, s in CANDIDATES]
		for future in as_completed(futures):
			future.result()

	rows = []
	for k, s in CANDIDATES:
		dev, test = parse(args.dataset, k, s)
		if dev is not None:
			rows.append([args.dataset, k, s, dev, test])
	with open(os.path.join(RESULT_DIR, 'grid_{}.csv'.format(args.dataset)), 'w', newline='', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(['dataset', 'n_layers', 'self_loop', 'NDCG@20(dev)', 'NDCG@20(test)'])
		writer.writerows(rows)
	for row in rows:
		print('{:<16}K={:<3}loop={:<3}dev={:<8}test={}'.format(row[0], row[1], row[2], row[3], row[4]))


if __name__ == '__main__':
	main()
