#!/usr/bin/env python3

import numpy as np
import io
from sklearn.cross_validation import KFold
from joblib import Parallel, delayed
import multiprocessing
import logging

logging.basicConfig(filename='paper.log',level=logging.DEBUG)

m = 6040 # users
n = 3952 # movies
MAXSCORE = 5

Rnan = np.full((m, n), np.nan) # Ratings matrix with nans instead of 0s
# Read the data into the rating matrix
with open('ml-1m/ratings.dat', 'r') as fp:
	for line in iter(fp.readline, ''):
		l = line.split('::')
		Rnan[int(l[0])-1,int(l[1])-1] = int(l[2])

def O_u(R, u_i): # item set of user u_i
	return np.nonzero(1-np.isnan(R[u_i,:]))
def U_o(R, o_j): # user set of item o_j
	return np.nonzero(1-np.isnan(R[:,o_j]))

def front_swap(M, inds, dim = 1):
	M_new = np.copy(M)
	swap_inds = [np.arange(inds.shape[0]) for i in range(dim)]
	orig_inds = [inds for i in range(dim)]
	temp = M_new[swap_inds]
	M_new[swap_inds] = M_new[orig_inds]
	M_new[orig_inds] = temp
	return M_new

def walk(N, n_users, P_star, alpha = 0.9):
	W = np.zeros((n_users,n_users), dtype = np.float) # The weight matrix for training set
	norm_P_star = P_star / (np.sum(P_star, axis= 1).reshape((n_users),1)) # Normalize the probabilities
	for r in range(N): # Do N runs for each training user
		users = np.arange(n_users) # Create the currently running users
		cur_users = np.copy(users) # The current user after starting from the running user itself.
		cont = np.random.rand(users.shape[0]) > alpha # Finish runs with alpha probability
		users = users[cont]
		while users.shape[0] > 0: # While there are currently running users
			for u in users: # Walk for each user
				u_new = np.random.choice(n_users, 1, p = norm_P_star[cur_users[u], :])[0] # Jump to a new user
				cur_users[u] = u_new
				W[u, u_new] += 1 # Increment the total number of visits to u_new starting from u
			cont = np.random.rand(users.shape[0]) > alpha # Finish runs with alpha probability
			users = users[cont]
		print('Walk '+str(r)+' done')
	return W / N # Calculate the average # of visits

class Predicter:
	def __init__(self, R, ind = None, alternative = False):
		self.R = R
		self.ind = ind
		self.r_bar_v = np.nanmean(self.R, axis = 1) # mean ratings of user u_i
		self.r_bar = np.nanmean(self.R) # global mean rating
		if alternative: # use our proposed method
			rl1 = np.log(R)+1
			self.P_uo = rl1 / np.nansum(rl1, axis = 1).reshape(R.shape[0],1)
			self.P_uo[np.isnan(self.P_uo)]= 0 # Type 1 walk, user to movie
		else:
			self.P_uo = (1 - np.isnan(self.R)) / np.sum(1 - np.isnan(self.R), axis = 1).reshape(self.R.shape[0],1) # Type 1 walk, user to movie

	def calc_r_hat(self, u_t, o_j, c_t):
		"""
		u_t -> target user
		o_j -> target movie
		c_t -> similarity vector of t to all users
		"""
		U_oj = U_o(self.R, o_j)
		with np.errstate(divide = 'ignore', invalid = 'ignore'):
			return self.r_bar_v[u_t] - self.r_bar + (np.nansum(c_t[U_oj]*self.R[U_oj, o_j]) / np.nansum(c_t[U_oj]))

	def sim(self, u_i): # similarity matrix from u_k to o_j, given u_i
		return MAXSCORE - np.absolute(self.R[u_i,:] - self.R)

	def P_ou(self, u_i):
		"""
		Transition probability matrix from movie to user, given
		a base user u_i. Note that axis 0 is still the user and
		the axis 1 is the movie.
		"""
		with np.errstate(divide='ignore', invalid='ignore'):
			s = self.sim(u_i)
			return s / np.nansum(s, axis = 0)

	def p(self, u_i):
		""" Transition probability from user u_i to each user. """
		return np.nansum(self.P_uo[u_i] * self.P_ou(u_i), axis = 1)

	def construct_P(self, filename):
		P = np.zeros((m,m))
		start = 0
		try:
			s = np.load(filename)
			start = s.shape[0]
			P[:start,:] = s
		except FileNotFoundError:
			pass
		for u_i in range(m)[start:]:
			P[u_i,:] = self.p(u_i)
			print('P '+str(u_i)+' done')
			if u_i % 100 == 0:
				np.save(filename, P[:u_i+1,:])
		return P

	def get_P(self):
		P = None
		file_name = 'P.npy' if self.ind is None else 'P'+str(self.ind)+'.npy'
		try:
			P = np.load(file_name)
			if P.shape[0] != m:
				P = self.construct_P(file_name)
		except FileNotFoundError:
			P = self.construct_P(file_name)
		return P

	def get_W(self, N, n_users, P_star, alpha = 0.9):
		W = None
		file_name = 'W.npy' if self.ind is None else 'W'+str(self.ind)+'.npy'
		try:
			W = np.load(file_name)
		except FileNotFoundError:
			W = walk(N, n_users, P_star, alpha = alpha)
			np.save(file_name, W)
		return W

	def get_C(self, W, pi, pi_self, test_set, alpha):
		C = None
		file_name = 'C.npy' if self.ind is None else 'C'+str(self.ind)+'.npy'
		try:
			C = np.load(file_name)
		except FileNotFoundError:
			size_ts = test_set.shape[0]
			C = np.vstack([front_swap(np.hstack((pi_self[k], alpha * np.sum(pi[k] * W.T, axis = 1))), \
				test_set, dim = 1) for k in range(size_ts)])
			np.save(file_name, C)
		return C

	def mean_absolute_error(self, R, C, test_set, held_out):
		maes = np.zeros(test_set.shape[0])
		for c_ind, u_i in enumerate(test_set):
			"""
			r_act = R[u_i, held_out[u_i]]
			ojs = np.arange(held_out.shape[1])[held_out[u_i]]
			r_hat = np.array([self.calc_r_hat(u_i, o_j, C[c_ind]) for o_j in ojs])
			maes[c_ind] = np.nanmean(np.absolute(r_act - r_hat))
			"""
			item_set = O_u(R, u_i)
			r_act = R[u_i, item_set]
			r_hat = np.round(np.array([self.calc_r_hat(u_i, o_j, C[c_ind]) for o_j in item_set[0]]))
			maes[c_ind] = np.nanmean(np.absolute(r_act - r_hat))
			print('MAE '+str(c_ind)+' done')
		return maes

	def get_MAE(self, R, C, test_set, held_out):
		MAE = None
		file_name = 'MAE.npy' if self.ind is None else 'MAE'+str(self.ind)+'.npy'
		try:
			MAE = np.load(file_name)
		except FileNotFoundError:
			MAE = self.mean_absolute_error(R, C, test_set, held_out)
			np.save(file_name, MAE)
		return MAE

class Tester:
	def __init__(self, n_folds):
		self.n_folds = n_folds
	def generate_test_sets(self):
		kfold = KFold(m, n_folds = self.n_folds, shuffle = True)
		i = 0
		l = []
		for train_set, test_set in kfold:
			np.save('TS'+str(i)+'.npy', test_set)
			l.append(test_set)
			i += 1
		return l
	def get_test_sets(self):
		test_sets = None
		try:
			test_sets = [np.load('TS'+str(i)+'.npy') for i in range(self.n_folds)]
		except FileNotFoundError:
			test_sets = self.generate_test_sets()
		return test_sets

	def generate_held_outs(self, test_sets, R, p = 0.9):
		l = []
		for i, test_set in enumerate(test_sets):
			ho = np.zeros((m,n), dtype = np.bool_)
			for test_user in test_set:
				items = O_u(R, test_user)[0]
				np.random.shuffle(items)
				items = items[:int(p*items.shape[0])]
				ho[test_user,items] = 1
			l.append(ho)
			np.save('HO'+str(i)+'.npy', ho)
		return l
	def get_held_outs(self, test_sets, R, p = 0.9):
		held_outs = None
		try:
			held_outs = [np.load('HO'+str(i)+'.npy') for i in range(self.n_folds)]
		except FileNotFoundError:
			held_outs = self.generate_held_outs(test_sets, R, p = p)
		return held_outs

	def test_loop(self, R, i, test_set, held_out, held_ratio = 0.9, alpha = 0.9):
		R_test = np.copy(R)
		R_test[held_out] = np.nan
		pred = Predicter(R_test, ind = i)
		P = pred.get_P()
		print('P'+str(i)+' ready')
		size_ts = test_set.shape[0]
		P_new = front_swap(P, test_set, dim = 2)
		P_star = P_new[size_ts:, size_ts:]
		pi_self = np.zeros((size_ts, size_ts))
		pi_self[np.diag_indices(size_ts)] = np.diag(P_new[:size_ts, :size_ts])
		pi = P_new[:size_ts, size_ts:]
		W = pred.get_W(m, m - size_ts, P_star, alpha = alpha)
		print('W'+str(i)+' ready')
		C = pred.get_C(W, pi, pi_self, test_set, alpha)
		MAE = pred.get_MAE(R, C, test_set, held_out)
		print('MAE'+str(i)+' ready')
		return MAE

	def parallel_test(self, R, held_ratio = 0.9, alpha = 0.9):
		test_sets = self.get_test_sets()
		held_outs = self.get_held_outs(test_sets, R, held_ratio)
		j = max([self.n_folds, multiprocessing.cpu_count()])
		maes = Parallel(n_jobs = 4)(delayed(self.test_loop)(R,i,\
			test_set,held_out, held_ratio, alpha) for \
		i, (test_set, held_out) in enumerate(zip(test_sets, held_outs)))
		return maes

	def test(self, R, held_ratio = 0.9, alpha = 0.9):
		test_sets = self.get_test_sets()
		held_outs = self.get_held_outs(test_sets, R, held_ratio)
		for i, (test_set, held_out) in enumerate(zip(test_sets, held_outs)):
			self.test_loop(R, i, test_set, held_out, held_ratio = held_ratio, alpha = alpha)

if __name__ == '__main__':
	with np.errstate(divide='raise', invalid='raise'):
		t = Tester(4)
		t.test(Rnan)