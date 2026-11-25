# -*- coding: UTF-8 -*-

""" Markov Graph Diffusion Collaborative Filtering (MGDCF)

Reference:
	"MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering"
	Jun Hu, Bryan Hooi, Shengsheng Qian, Quan Fang, Changsheng Xu. TKDE, 2022.
	Paper: https://arxiv.org/abs/2204.02338
Official code: https://github.com/hujunxianligong/MGDCF

This file re-implements the *Hetero-MGDCF* (Hetero-MGDN encoder + InfoBPR loss)
inside ReChorus, following the paper's Equation (3)-(8) and (22):

	H(0) = X
	H(k) = beta * A_hat H(k-1) + alpha * H(0),   k = 1, ..., K
	Z    = H(K) / Gamma,   Gamma = beta^K + sum_{k=0}^{K-1} alpha * beta^k
	A_hat = D^-1/2 (A + I) D^-1/2                 (Eq.3, self-loops are kept)

Loss:
	InfoBPR = -log( exp(s_pos) / (exp(s_pos) + sum_k exp(s_neg_k)) )   (Eq.22)
	with --num_neg negative samples. num_neg == 1 degenerates to BPR.

Special cases covered by the same code (used as ablations):
	--n_layers 0                 -> MF-InfoBPR (Eq.27, K = 0 gives Z = X)
	--loss_type BPR --alpha .1 --beta .9 --n_layers 4
	                             -> APPNP (i.e., MGDN optimized with BPR)

CMD example:
	python main.py --model_name MGDCF --emb_size 64 --n_layers 4 --alpha 0.1 --beta 0.9 \
		--loss_type InfoBPR --num_neg 8 --lr 1e-3 --l2 0 --batch_size 2048 \
		--dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp

from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel
from utils.layers import SparseMM


class MGDCFBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--n_layers', type=int, default=4,
							help='Number of Markov transition steps K (0 for MF).')
		parser.add_argument('--alpha', type=float, default=0.1,
							help='Weight of the initial representation H(0) (teleport probability).')
		parser.add_argument('--beta', type=float, default=0.9,
							help='Weight of the propagated representation.')
		parser.add_argument('--loss_type', type=str, default='InfoBPR',
							help='Ranking loss: InfoBPR or BPR.')
		parser.add_argument('--self_loop', type=int, default=1,
							help='Whether to add self-loops in the affinity matrix (Eq.3).')
		return parser

	@staticmethod
	def build_adjmat(user_count, item_count, train_mat, self_loop=True):
		"""Build D^-1/2 A D^-1/2 for the heterogeneous user-item graph."""
		rows, cols = list(), list()
		for user in train_mat:
			for item in train_mat[user]:
				rows.append(user)
				cols.append(item)
		rows, cols = np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)
		R = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
						  shape=(user_count, item_count))
		adj = sp.bmat([[None, R], [R.T, None]], format='csr', dtype=np.float32)
		if self_loop:
			adj = adj + sp.eye(adj.shape[0], format='csr', dtype=np.float32)

		rowsum = np.asarray(adj.sum(axis=1)).flatten()
		d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
		nz = rowsum > 0
		d_inv_sqrt[nz] = np.power(rowsum[nz], -0.5)
		D_inv_sqrt = sp.diags(d_inv_sqrt)
		norm_adj = (D_inv_sqrt @ adj @ D_inv_sqrt).tocoo()
		return norm_adj

	def _base_init(self, args, corpus):
		self.emb_size = args.emb_size
		self.n_layers = args.n_layers
		self.alpha = args.alpha
		self.beta = args.beta
		self.loss_type = args.loss_type
		self.self_loop = args.self_loop
		self.gamma = self.beta ** self.n_layers + \
			sum([self.alpha * self.beta ** k for k in range(self.n_layers)])
		if self.self_loop:
			norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items,
										 corpus.train_clicked_set, self_loop=True)
		else:
			norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items,
										 corpus.train_clicked_set, self_loop=False)
		self._base_define_params(norm_adj)
		self.apply(self.init_weights)

	def _base_define_params(self, norm_adj):
		# CSR SpMM is roughly 50x faster than COO on CPU; the transpose is
		# precomputed so that the backward pass is also a fast CSR-dense product.
		csr = norm_adj.tocsr()
		csr_t = csr.T.tocsr()
		self.sparse_norm_adj = self._to_csr_tensor(csr)
		self.sparse_norm_adj_t = self._to_csr_tensor(csr_t)
		self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
		self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

	def _to_csr_tensor(self, csr):
		return torch.sparse_csr_tensor(
			torch.from_numpy(csr.indptr.astype(np.int64)),
			torch.from_numpy(csr.indices.astype(np.int64)),
			torch.from_numpy(csr.data.astype(np.float32)),
			size=csr.shape).to(self.device)

	def _propagate(self):
		"""Markov graph diffusion (Eq.6-8). Returns the full (user+item) representations."""
		x0 = torch.cat([self.u_embeddings.weight, self.i_embeddings.weight], dim=0)
		h = x0
		for _ in range(self.n_layers):
			h = self.beta * SparseMM.apply(self.sparse_norm_adj, self.sparse_norm_adj_t, h) + self.alpha * x0
		return h / self.gamma

	def forward(self, feed_dict):
		self.check_list = []
		u_ids = feed_dict['user_id']  # [batch_size]
		i_ids = feed_dict['item_id']  # [batch_size, -1]

		all_embeddings = self._propagate()
		u_vectors = all_embeddings[:self.user_num][u_ids]
		i_vectors = all_embeddings[self.user_num:][i_ids]

		prediction = (u_vectors[:, None, :] * i_vectors).sum(dim=-1)  # [batch_size, -1]
		return {'prediction': prediction.view(feed_dict['batch_size'], -1)}

	def loss(self, out_dict):
		predictions = out_dict['prediction']
		pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
		if self.loss_type == 'InfoBPR':
			# Eq.22: multi-class (InfoNCE style) softmax over 1 positive + N negatives
			loss = -torch.log_softmax(predictions, dim=1)[:, 0].mean()
		elif self.loss_type == 'BPR':
			loss = -(pos_pred[:, None] - neg_pred).sigmoid().log().mean()
		else:
			raise ValueError('Undefined loss type: {}'.format(self.loss_type))
		return loss


class MGDCF(GeneralModel, MGDCFBase):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'n_layers', 'alpha', 'beta', 'loss_type', 'num_neg', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = MGDCFBase.parse_model_args(parser)
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		GeneralModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		return MGDCFBase.forward(self, feed_dict)

	def loss(self, out_dict):
		return MGDCFBase.loss(self, out_dict)


class MGDCFImpression(ImpressionModel, MGDCFBase):
	reader = 'ImpressionReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'n_layers', 'alpha', 'beta', 'loss_type', 'num_neg', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = MGDCFBase.parse_model_args(parser)
		return ImpressionModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		return MGDCFBase.forward(self, feed_dict)

	def loss(self, out_dict):
		return MGDCFBase.loss(self, out_dict)
