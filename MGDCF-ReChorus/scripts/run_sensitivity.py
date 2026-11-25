# -*- coding: UTF-8 -*-

""" Hyper-parameter sensitivity of Hetero-MGDCF (one factor at a time).

	python scripts/run_sensitivity.py --dataset Grocery_and_Gourmet_Food --jobs 6
Writes ../results/sensitivity.csv
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
LOG_DIR = os.path.join(ROOT, 'log', 'sensitivity')
RESULT_DIR = os.path.join(ROOT, 'results')

GRID = {
	'alpha': [0.05, 0.1, 0.3, 0.5],
	'beta': [0.7, 0.9, 0.99],
	'n_layers': [1, 2, 3, 4],
}
BASE = {'alpha': 0.1, 'beta': 0.9, 'n_layers': 2}
METRIC_RE = re.compile(r'NDCG@20:([\d.]+)')


def log_path(param, value, dataset):
	return os.path.join(LOG_DIR, 'MGDCF__{}__{}_{}__{}.txt'.format(dataset, param, value, dataset))


def run_one(param, value, dataset, epoch):
	cfg = dict(BASE)
	cfg[param] = value
	cmd = [sys.executable, 'main.py', '--model_name', 'MGDCF', '--dataset', dataset,
		   '--path', '../data/', '--random_seed', '0', '--epoch', str(epoch),
		   '--log_file', log_path(param, value, dataset),
		   '--emb_size', '64', '--n_layers', str(cfg['n_layers']),
		   '--alpha', str(cfg['alpha']), '--beta', str(cfg['beta']),
		   '--loss_type', 'InfoBPR', '--num_neg', '8', '--lr', '1e-3', '--l2', '0',
		   '--batch_size', '512' if dataset == 'Grocery_and_Gourmet_Food' else '2048',
		   '--num_workers', '0', '--save_final_results', '0',
		   '--topk', '20', '--metric', 'NDCG,HR', '--main_metric', 'NDCG@20', '--early_stop', '5']
	env = dict(os.environ)
	env['OMP_NUM_THREADS'] = '6'
	env['MKL_NUM_THREADS'] = '6'
	print('[START] {}={}'.format(param, value), flush=True)
	with open(log_path(param, value, dataset), 'w', encoding='utf-8') as f:
		subprocess.run(cmd, cwd=SRC, stdout=f, stderr=subprocess.STDOUT, env=env)
	print('[DONE ] {}={}'.format(param, value), flush=True)


def parse(param, value, dataset):
	path = log_path(param, value, dataset)
	if not os.path.exists(path):
		return None
	ndcg = None
	with open(path, encoding='utf-8', errors='ignore') as f:
		for line in f:
			if 'Test After Training' in line:
				m = METRIC_RE.search(line)
				if m:
					ndcg = float(m.group(1))
	return ndcg


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food')
	parser.add_argument('--epoch', type=int, default=60)
	parser.add_argument('--jobs', type=int, default=6)
	args = parser.parse_args()

	os.makedirs(LOG_DIR, exist_ok=True)
	os.makedirs(RESULT_DIR, exist_ok=True)
	tasks = [(p, v) for p, values in GRID.items() for v in values]
	with ThreadPoolExecutor(max_workers=args.jobs) as pool:
		futures = [pool.submit(run_one, p, v, args.dataset, args.epoch) for p, v in tasks]
		for future in as_completed(futures):
			future.result()

	rows = []
	for p, v in tasks:
		ndcg = parse(p, v, args.dataset)
		if ndcg is not None:
			rows.append([args.dataset, p, v, ndcg])
	with open(os.path.join(RESULT_DIR, 'sensitivity.csv'), 'w', newline='', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(['dataset', 'param', 'value', 'NDCG@20(test)'])
		writer.writerows(rows)
	for row in rows:
		print('{:<28}{:<10}{:<8}{:.4f}'.format(*row))


if __name__ == '__main__':
	main()
