import ArgsParser
import logging
import os
from datetime import datetime
from utils import setup_logging

# If the GPU is based on the nvdia Ampere architecture uncomment this line as it speed-up training up to 3x reducing memory footprint
# torch.backends.cuda.matmul.allow_tf32 = True


# Initial setup: parser, logging...
args = ArgsParser.parse_arguments()
start_time = datetime.now()
args.output_dir = os.path.join("logs", args.output_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))

setup_logging(args.output_dir, "info")
make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {cpu_count()} CPUs")

model_name = args.model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)