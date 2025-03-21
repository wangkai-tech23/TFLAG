import argparse

parser = argparse.ArgumentParser(description='Denoise')
parser.add_argument('--dir_data', type=str, default='./dataset')
parser.add_argument('--data_set', type=str, default='cadets', choices=('theia', 'clearscope','cadets'))


parser.add_argument('--anomaly_alpha', type=float, default=5e-3, help="gnn anomaly loss param")
parser.add_argument('--memory_size', type=int, default=5000, help="gdn memory_size")
parser.add_argument('--sample_size', type=int, default=2000, help="gdn sample_size")
parser.add_argument('--mask_label', action='store_true', default=False)
parser.add_argument('--mask_ratio', type=float, default=0.5)
##data param
parser.add_argument('--n_neighbors', type=int, default=20, help='Maximum number of connected edge per node')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_epochs', type=int, default=5)
parser.add_argument('--num_data_workers', type=int, default=8)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--accelerator', type=str, default='ddp')

##model param
parser.add_argument('--ckpt_file', type=str, default='./')
parser.add_argument('--input_node_dim', type=int, default=3)
parser.add_argument('--input_edge_dim', type=int, default=32)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--n_heads', type=int, default=2)
parser.add_argument('--drop_out', type=float, default=0.2)
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--learning_rate', type=float, default=0.0001)

args = parser.parse_args()

