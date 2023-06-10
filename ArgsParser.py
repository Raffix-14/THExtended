import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="THExtended - Extractive Summarization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=8, help="Train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Steps before gradient update. More = slower training but larger batch size")
    parser.add_argument("--epochs_num", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="If a name is provided, the dataset will be downloaded from HuggingFace. "
                             "If a path is provided, the dataset will be loaded from the specified path.")
    parser.add_argument("--num_train_examples", type=int, default=3000, help="Number of training documents")
    parser.add_argument("--num_val_examples", type=int, default=500, help="Number of evaluation documents")
    parser.add_argument("--num_test_examples", type=int, default=500, help="Number of test documents")

    parser.add_argument("--model_name_or_path", type=str, default='bert-base-uncased',
                        help='Path to pretrained encoder model or model identifier from huggingface.co/models')
    parser.add_argument("--output_dir", type=str, default="out_tmp", help="Path to save checkpoints and logs")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for the AdamW optimizer")
    parser.add_argument("--seed", type=int, default=14, help="Random seed")
    parser.add_argument("--save_dataset_on_disk", type=bool, default=False)
    parser.add_argument("--push_to_hub", type=bool, default=False,
                        help="Push model checkpoints to the Hub if set to true")
    ########################################
    args = parser.parse_args()

    """parser.add_argument("--criterion", type=str, default='triplet', help='loss to be used',
                        choices=["triplet", "sare_ind", "sare_joint"])
    parser.add_argument("--trunc_te", type=int, default=None, choices=list(range(0, 14)))"""
    return args
