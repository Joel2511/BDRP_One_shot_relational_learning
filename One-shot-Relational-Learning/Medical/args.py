import argparse

def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/gpfs/workdir/anilj/Medical/data", type=str)
    parser.add_argument("--embed_dim", default=100, type=int)
    parser.add_argument("--few", default=1, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--neg_num", default=50, type=int)
    parser.add_argument("--random_embed", action='store_true')
    parser.add_argument("--train_few", default=1, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--margin", default=1.0, type=float)
    parser.add_argument("--max_batches", default=5000, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--process_steps", default=2, type=int)
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--eval_every", default=1000, type=int)
    parser.add_argument("--fine_tune", action='store_true')
    parser.add_argument("--aggregate", default='max', type=str)
    parser.add_argument("--max_neighbor", default=100, type=int)
    parser.add_argument("--no_meta", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--grad_clip", default=5.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--embed_model", default='ComplEx', type=str)
    parser.add_argument("--prefix", default='medical_gating_v1', type=str)
    parser.add_argument("--seed", default=19940419, type=int)

    # --- GATING & SEMANTIC HYPERPARAMETERS ---
    parser.add_argument("--knn_k", default=10, type=int, help="k-NN neighbors for topk selection")
    parser.add_argument("--use_semantic", action='store_true', help="Integrate semantic embeddings channel")
    parser.add_argument("--semantic_type", type=str, default='pubmedbert',
                        choices=['sapbert', 'pubmedbert', 'luke', 'bert', 'roberta'])
    parser.add_argument("--gate_lr", default=0.001, type=float, help="Learning rate for gating layer")

    # gate_mode and gate_alpha are set automatically from DATASET_CONFIGS via prefix.
    # These CLI args are provided as override escape hatches — normally leave unset.
    parser.add_argument("--gate_mode", default=None, type=str,
                        choices=['learned', 'fixed'],
                        help="Override dataset-config gate mode. Leave unset to use config default.")
    parser.add_argument("--gate_alpha", default=None, type=float,
                        help="Override fixed gate alpha (structural weight). Only used when gate_mode=fixed.")

    # --- DATASET FILTERING ---
    parser.add_argument('--object_only', action='store_true',
                        help='Use filtered Object-Property only data (Medical)')

    args = parser.parse_args()
    args.save_path = 'models/' + args.prefix

    print("HYPERPARAMETERS")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))
    print("----------------------------")
    return args
