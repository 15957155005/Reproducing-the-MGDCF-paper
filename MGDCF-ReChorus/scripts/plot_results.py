# -*- coding: UTF-8 -*-

""" Plot the experiment figures used in the report.

	python scripts/plot_results.py
Outputs to ../results/figures/*.png
"""

import os
import re
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'log')
RESULT_DIR = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(RESULT_DIR, 'figures')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CONFIGS = ['BPRMF', 'NeuMF', 'LightGCN', 'APPNP', 'MF-InfoBPR', 'Hetero-MGDCF']
LABELS = {'BPRMF': 'BPRMF', 'NeuMF': 'NeuMF', 'LightGCN': 'LightGCN',
		  'APPNP': 'APPNP (MGDN+BPR)', 'MF-InfoBPR': 'MF-InfoBPR', 'Hetero-MGDCF': 'Hetero-MGDCF'}
MARKERS = {'BPRMF': 'o', 'NeuMF': 's', 'LightGCN': '^', 'APPNP': 'v',
		   'MF-InfoBPR': 'D', 'Hetero-MGDCF': '*'}
EPOCH_RE = re.compile(r'^Epoch (\d+)\s+loss=([\d.]+).*NDCG@20:([\d.]+)')


def read_curve(config, dataset, seed=0):
	path = os.path.join(LOG_DIR, '{}__{}__seed{}.txt'.format(config, dataset, seed))
	epochs, ndcgs = [], []
	if not os.path.exists(path):
		return epochs, ndcgs
	with open(path, encoding='utf-8', errors='ignore') as f:
		for line in f:
			m = EPOCH_RE.match(line.strip())
			if m:
				epochs.append(int(m.group(1)))
				ndcgs.append(float(m.group(3)))
	return epochs, ndcgs


def plot_convergence(dataset):
	plt.figure(figsize=(6.4, 4.2))
	for config in CONFIGS:
		epochs, ndcgs = read_curve(config, dataset)
		if len(epochs):
			plt.plot(epochs, ndcgs, label=LABELS[config], marker=MARKERS[config], markersize=3, linewidth=1.4)
	plt.xlabel('训练轮数 (epoch)')
	plt.ylabel('NDCG@20 (验证集)')
	plt.title('{} 上各模型的收敛曲线'.format(dataset.replace('_', '-')))
	plt.grid(alpha=0.3)
	plt.legend(fontsize=8)
	plt.tight_layout()
	path = os.path.join(FIG_DIR, 'convergence_{}.png'.format(dataset))
	plt.savefig(path, dpi=200)
	plt.close()
	print('saved', path)


def plot_comparison(dataset):
	rows = []
	with open(os.path.join(RESULT_DIR, 'summary.csv'), encoding='utf-8') as f:
		for row in csv.DictReader(f):
			if row['dataset'] == dataset and row['NDCG@20(test)']:
				rows.append((row['model'], float(row['NDCG@20(test)']), float(row['HR@20(test)'])))
	rows.sort(key=lambda x: x[1])
	models = [LABELS.get(r[0], r[0]) for r in rows]
	ndcg = [r[1] for r in rows]
	hr = [r[2] for r in rows]

	fig, axes = plt.subplots(1, 2, figsize=(10, 4))
	for ax, values, title in zip(axes, [ndcg, hr], ['NDCG@20 (测试集)', 'HR@20 (测试集)']):
		bars = ax.barh(models, values, color='#4C72B0')
		ax.set_title(title)
		ax.grid(axis='x', alpha=0.3)
		for bar, value in zip(bars, values):
			ax.text(value, bar.get_y() + bar.get_height() / 2, ' {:.4f}'.format(value),
					va='center', fontsize=8)
		ax.set_xlim(0, max(values) * 1.18)
	fig.suptitle('{} 上的性能对比'.format(dataset.replace('_', '-')))
	plt.tight_layout()
	path = os.path.join(FIG_DIR, 'comparison_{}.png'.format(dataset))
	plt.savefig(path, dpi=200)
	plt.close()
	print('saved', path)


def plot_sensitivity(csv_name, dataset, title):
	path = os.path.join(RESULT_DIR, csv_name)
	if not os.path.exists(path):
		return
	data = {}
	with open(path, encoding='utf-8') as f:
		for row in csv.DictReader(f):
			if row['dataset'] != dataset:
				continue
			data.setdefault(row['param'], []).append((float(row['value']), float(row['NDCG@20(test)'])))

	params = list(data.keys())
	fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 3.6))
	if len(params) == 1:
		axes = [axes]
	for ax, param in zip(axes, params):
		pairs = sorted(data[param])
		ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker='o', color='#DD8452')
		ax.set_xlabel(param)
		ax.set_ylabel('NDCG@20 (测试集)')
		ax.set_title(param)
		ax.grid(alpha=0.3)
	fig.suptitle(title)
	plt.tight_layout()
	out = os.path.join(FIG_DIR, 'sensitivity_{}.png'.format(dataset))
	plt.savefig(out, dpi=200)
	plt.close()
	print('saved', out)


if __name__ == '__main__':
	os.makedirs(FIG_DIR, exist_ok=True)
	datasets = ['Grocery_and_Gourmet_Food', 'MovieLens_1M']
	for dataset in datasets:
		plot_convergence(dataset)
		if os.path.exists(os.path.join(RESULT_DIR, 'summary.csv')):
			plot_comparison(dataset)
	plot_sensitivity('sensitivity.csv', 'Grocery_and_Gourmet_Food', 'Hetero-MGDCF 超参数敏感性 (Grocery)')
