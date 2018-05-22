import pylab as plt
import matplotlib
import re
import numpy as np


class OOMFormatter(plt.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.0f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        plt.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

matplotlib.rcParams.update({'font.size': 18})
log_files = ['wgan-gp.txt', 'wc-wgan-gp.txt', 'sngan.txt', 'wc-sngan.txt']
color = ['red', 'red', 'blue', 'blue']
linestyle = ['dashed', 'solid', 'dashed', 'solid']
labels = ['GP', 'WC GP', 'SN', 'WC SN']


for score_type in ('fid', 'inception'):
	pattern = ".*INCEPTION SCORE: (.*), .*" if score_type == 'inception' else  ".*FID SCORE: (.*)"

	axis = plt.figure().add_subplot(111)

	for i, f in enumerate(log_files):
	    score = []
	    for line in open(f):
		is_match = re.match(pattern, line)
		if is_match is not None:
		    score.append(float(is_match.group(1)))
	    x = np.arange(3, len(score))
	    score = score[3:]
	    plt.plot(x * 1000, np.array(score), color=color[i], linestyle=linestyle[i], label=labels[i])

	axis.set_xlabel("Iterations")
	axis.set_ylabel("Inception Scores" if score_type == 'inception' else "FID Scores 10k")
	axis.legend()
	axis.yaxis.set_ticks_position('both')
	plt.tick_params(labelright=True)

	fmt = OOMFormatter(order=3)

	axis.xaxis.set_major_formatter(fmt)

	plt.savefig('inception_score.pdf' if score_type == 'inception' else "fid_score.pdf", bbox_inches='tight')




