import argparse
from gans import Generator, Discriminator

parser = argparse.ArgumentParser(description='Parameters to use in GAN train process')

parser.add_argument('--cuda', type=str, default='False', required=True, help='')
parser.add_argument('--data_path', type=str, default='~/Data/mnist', required=True, help='')
parser.add_argument('--out_path', type=str, default='output', required=True, help='')
parser.add_argument('--batch_size', type=int, default=508, required=True, help='')
parser.add_argument('--image_channels', type=int, default=1, required=True, help='')
parser.add_argument('--z_dim', type=int, default=100, required=True, help='')
parser.add_argument('--generator_hidden_layer_size', type=str, default=64, required=True, help='')
parser.add_argument('--x_dim', type=int, default=64, required=True, help='')
parser.add_argument('--discriminator_hidden_layer_size', type=int, default=64, required=True, help='')
parser.add_argument('--epochs', type=int, default=6, required=True, help='')
parser.add_argument('--real_label', type=int, default=1, required=True, help='')
parser.add_argument('--fake_label', type=int, default=0, required=True, help='')
parser.add_argument('--learning_rate', type=float, default=2e-4, required=True, help='')
parser.add_argument('--seed', type=int, default=1, required=True, help='')

args = parser.parse_args()
print(args.cuda)
