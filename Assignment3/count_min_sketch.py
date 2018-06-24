from array import array
from random import randint
from math import log, e, ceil
from itertools import izip

class CountMinSketch(object):
	def __init__(self, w=None, d=None, delta=None, epsilon=None):
		
		if w is not None and d is not None:
			self.w = w
			self.d = d
		elif delta is not None and epsilon is not None:
			self.w = int(ceil(e/epsilon))
			self.d = int(ceil(log(1./delta)))
			print self.w, self.d
		else:
			raise Exception("You must either supply both w and d or delta and epsilon.")
		
		self.counts = [array('L', (0 for _ in xrange(self.w))) for _ in xrange(self.d)]
		upper_bound = 2147483647
		step = upper_bound / (self.d-1)
		ranges = [(i*step, step*(i+1)-1) for i in xrange(self.d-1)]
		self.mask = array('L', (randint(low, high) for low, high in ranges))
		
	def get_columns(self, a):
		h = hash(a)
		w = self.w
		
		yield h % w
		for m in self.mask:
			yield (h ^ m) % w
			
		
	def update(self, a, val=1):
		for row, col in izip(self.counts, self.get_columns(a)):
		  row[col] += val
	
	def query(self, a):
		return min(row[col] for row, col in izip(self.counts, self.get_columns(a)))
	
	def __getitem__(self, a):
		return self.query(a)
	
	def __setitem__(self, a, val):
		for row, col in izip(self.counts, self.get_columns(a)):
		  row[col] = val
