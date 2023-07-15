import argparse

"""
    This file contains the arguments parser for the THExtended project.
    The arguments are divided into two categories: training and testing.
    The training arguments are used to train the model, while the testing arguments are used to test the model.
    The arguments are parsed using the argparse library.
"""

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
    parser.add_argument("--num_highlights", type=int, default=3,
                        help="Number of highlights to extract and use for testing")
    parser.add_argument("--model_name_or_path", type=str, default='bert-base-uncased',
                        help='Path to pretrained encoder model or model identifier from huggingface.co/models')
    parser.add_argument("--output_dir", type=str, default="out_tmp", help="Path to save checkpoints and logs")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for the AdamW optimizer")
    parser.add_argument("--alpha", type=float, default=1.0, help="Penalty weight for the semantic-aware loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dataset_on_disk", type=int, default=0, help="Save the processed dataset on disk if set to 1")
    parser.add_argument("--push_to_hub", type=int, default=0, help="Push model checkpoints to the Hub if set to 1")
    parser.add_argument("--trigram_blocking", type=int, default=0, help="Apply trigram blocking technique during testing if set to 1")

    args = parser.parse_args()
    return args
