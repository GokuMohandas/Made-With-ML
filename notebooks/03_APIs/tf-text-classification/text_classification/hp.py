import os
import sys
sys.path.append(".")
from argparse import ArgumentParser

from text_classification import config

if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--exp-cmd', type=str,
                        required=True, help="experiment command")
    parser.add_argument('--embedding-dim-list', nargs='+',
                        default=[50, 100], type=int, help="embedding dims")
    parser.add_argument('--num-filters-list', nargs='+',
                        default=[50, 100], type=int, help="num filters")
    parser.add_argument('--hidden-dim-list', nargs='+',
                        default=[64, 128], type=int, help="hidden dims")
    parser.add_argument('--dropout-p-list', nargs='+',
                        default=[0.1], type=float, help="dropouts")
    parser.add_argument('--learning-rate-list', nargs='+',
                        default=[1e-3, 1e-4], type=float, help="lr rates")
    args = parser.parse_args()

    # Run experiments
    experiment_num = 1
    num_experiments = len(args.embedding_dim_list) * \
        len(args.num_filters_list) * \
        len(args.hidden_dim_list) * \
        len(args.dropout_p_list) * \
        len(args.learning_rate_list)
    for embedding_dim in args.embedding_dim_list:
        for num_filters in args.num_filters_list:
            for hidden_dim in args.hidden_dim_list:
                for dropout_p in args.dropout_p_list:
                    for learning_rate in args.learning_rate_list:
                        experiment_cmd = args.exp_cmd + \
                            f' --embedding-dim {embedding_dim}' \
                            f' --num-filters {num_filters}' \
                            f' --hidden-dim {hidden_dim}' \
                            f' --dropout-p {dropout_p}' \
                            f' --learning-rate {learning_rate}'
                        config.logger.info(
                            f"→ Running experiment {experiment_num}/{num_experiments}:\n{experiment_cmd}")
                        os.system(experiment_cmd)
                        experiment_num += 1
