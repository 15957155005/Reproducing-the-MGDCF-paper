# -*- coding: UTF-8 -*-

""" Batch runner for the MGDCF reproduction experiments.

It launches `src/main.py` for a list of model/dataset configurations (with a
configurable degree of parallelism), then parses the logs and writes a summary
table to ../results/.

Usage:
	python scripts/run_experiments.py --datasets Grocery_and_Gourmet_Food MovieLens_1M --jobs 6
	python scripts/run_experiments.py --datasets Grocery_and_Gourmet_Food --only MGDCF
	python scripts/run_experiments.py --summary          # only re-parse existing logs
"""

import os
import re
import sys
import csv
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
LOG_DIR = os.path.join(ROOT, 'log')
RESULT_DIR = os.path.join(ROOT, 'results')

# config name -> (model_name, extra args)
MODELS = {
	'BPRMF': ('BPRMF', ['--emb_size', '64', '--lr', '1e-3', '--l2', '1e-6', '--batch_size', '256']),
	'LightGCN': ('LightGCN', ['--emb_size', '64', '--n_layers', '3', '--lr', '1e-3', '--l2', '1e-8',
							  '--batch_size', '2048']),
	'NeuMF': ('NeuMF', ['--emb_size', '64', '--layers', '[64]', '--lr', '5e-4', '--l2', '1e-7',
						'--dropout', '0.2', '--batch_size', '256']),
	# APPNP is exactly a Hetero-MGDN trained with the (single negative) BPR loss
	'APPNP': ('MGDCF', ['--emb_size', '64', '--n_layers', '2', '--alpha', '0.1', '--beta', '0.9',
						'--loss_type', 'BPR', '--num_neg', '1', '--lr', '1e-3', '--l2', '0',
						'--batch_size', '2048']),
	# MF is the K = 0 special case of MGDN (Eq.27 of the paper)
	'MF-InfoBPR': ('MGDCF', ['--emb_size', '64', '--n_layers', '0', '--loss_type', 'InfoBPR',
							 '--num_neg', '8', '--lr', '1e-3', '--l2', '0', '--batch_size', '2048']),
	'Hetero-MGDCF': ('MGDCF', ['--emb_size', '64', '--n_layers', '2', '--alpha', '0.1', '--beta', '0.9',
							   '--loss_type', 'InfoBPR', '--num_neg', '8', '--lr', '1e-3', '--l2', '0',
							   '--batch_size', '2048']),
}

COMMON = ['--num_workers', '0', '--save_final_results', '0', '--topk', '20',
		  '--metric', 'NDCG,HR', '--main_metric', 'NDCG@20', '--early_stop', '10']

# Grocery only has ~120k training interactions, so a 2048 batch would give only 60
# gradient steps per epoch; a smaller batch keeps the number of updates comparable
# to the baselines. ML-1M follows the batch size used in the paper.
BATCH_SIZE = {'Grocery_and_Gourmet_Food': '512', 'MovieLens_1M': '2048'}


def log_path(config, dataset, seed):
	return os.path.join(LOG_DIR, '{}__{}__seed{}.txt'.format(config, dataset, seed))


def run_one(config, dataset, seed, epoch):
	model_name, extra = MODELS[config]
	extra = list(extra)
	if '--batch_size' in extra:
		extra[extra.index('--batch_size') + 1] = BATCH_SIZE.get(dataset, '2048')
	cmd = [sys.executable, 'main.py', '--model_name', model_name, '--dataset', dataset,
		   '--path', '../data/', '--random_seed', str(seed), '--epoch', str(epoch),
		   '--log_file', log_path(config, dataset, seed)] + extra + COMMON
	env = dict(os.environ)
	env['OMP_NUM_THREADS'] = '4'
	env['MKL_NUM_THREADS'] = '4'
	print('[START] {} / {}'.format(config, dataset), flush=True)
	with open(log_path(config, dataset, seed), 'w', encoding='utf-8') as f:
		proc = subprocess.run(cmd, cwd=SRC, stdout=f, stderr=subprocess.STDOUT, env=env)
	print('[DONE ] {} / {} (exit {})'.format(config, dataset, proc.returncode), flush=True)
	return config, dataset, proc.returncode


METRIC_RE = re.compile(r'([A-Za-z]+@\d+):([\d.]+)')


def parse_log(path):
	dev, test = {}, {}
	if not os.path.exists(path):
		return None
	with open(path, encoding='utf-8', errors='ignore') as f:
		for line in f:
			if 'Dev  After Training' in line:
				dev = {k: float(v) for k, v in METRIC_RE.findall(line)}
			elif 'Test After Training' in line:
				test = {k: float(v) for k, v in METRIC_RE.findall(line)}
	if not test:
		return None
	return {'dev': dev, 'test': test}


def collect(datasets, seeds):
	os.makedirs(RESULT_DIR, exist_ok=True)
	rows = []
	for dataset in datasets:
		for config in MODELS:
			for seed in seeds:
				res = parse_log(log_path(config, dataset, seed))
				if res is None:
					continue
				rows.append([dataset, config, seed,
							 res['test'].get('NDCG@20', ''), res['test'].get('HR@20', '')])
	with open(os.path.join(RESULT_DIR, 'summary.csv'), 'w', newline='', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(['dataset', 'model', 'seed', 'NDCG@20(test)', 'HR@20(test)'])
		writer.writerows(rows)
	print('\n{:<28}{:<16}{:<8}{:<12}{}'.format('dataset', 'model', 'seed', 'NDCG@20', 'HR@20'))
	for row in rows:
		ndcg = '{:.4f}'.format(row[3]) if row[3] != '' else '-'
		hr = '{:.4f}'.format(row[4]) if row[4] != '' else '-'
		print('{:<28}{:<16}{:<8}{:<12}{}'.format(row[0], row[1], row[2], ndcg, hr))
	return rows


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--datasets', nargs='+', default=['Grocery_and_Gourmet_Food', 'MovieLens_1M'])
	parser.add_argument('--seeds', nargs='+', type=int, default=[0])
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--jobs', type=int, default=4)
	parser.add_argument('--only', nargs='+', default=None, help='subset of config names')
	parser.add_argument('--summary', action='store_true', help='only parse the existing logs')
	args = parser.parse_args()

	configs = [c for c in MODELS if args.only is None or c in args.only]
	os.makedirs(LOG_DIR, exist_ok=True)

	if not args.summary:
		tasks = [(c, d, s) for d in args.datasets for c in configs for s in args.seeds]
		with ThreadPoolExecutor(max_workers=args.jobs) as pool:
			futures = [pool.submit(run_one, c, d, s, args.epoch) for c, d, s in tasks]
			for future in as_completed(futures):
				future.result()
	collect(args.datasets, args.seeds)


if __name__ == '__main__':
	main()
