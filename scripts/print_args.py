import argparse
import torch

"""
Tiny utility to print the command-line args used for a checkpoint
"""

parser = argparse.ArgumentParser()
parser.add_argument('--model_num')


def main(args):
	model_list = args.model_num.split(',')
	for q in model_list:
		# model_num = args.model_num.split(',')[q]
		checkpoint = torch.load('../checkpoint/checkpoint_{}.pt'.format(q), map_location='cpu')
		# p = './args.txt'
		# fa = open(p, "w")
		for k, v in checkpoint['args'].items():
			# fa.write('{}  :  {}. \n'.format(k, v))
			print('{}  :  {}. \n'.format(k, v))
		# fa.close()

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
