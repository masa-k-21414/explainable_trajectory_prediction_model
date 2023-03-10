import argparse
import torch
import shutil
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model_num')


def main(args):
	for d in args.model_num.split(','):
		print('[Attention!] model {} will be deleted!!'.format(d))
		a = input("[y/n]") 
		if a == 'y':
			try:
				shutil.rmtree('./logger/checkpoint_{}_t'.format(d))
				print('logger is deleted')
			except FileNotFoundError:
				print('logger is not found.')
			
			try:
				shutil.rmtree('./video/checkpoint_{}_v'.format(d))
				print('video is deleted')
			except FileNotFoundError:
				print('video is not found.')
		else:
			print('model {} is NOT deleted.'.format(d))

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
