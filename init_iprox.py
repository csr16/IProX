# ```python
import argparse
import os
import sys
import torch
import logging
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from utils.get_training_dataset import get_training_dataset
from utils.init_with_ipsvd import init_proxy_model_with_IPSVD, load_proxy_model, save_proxy_model
from utils.grad_align import train_with_gradient_alignment, get_target_layer_pairs
from utils.util import llama3_2_chat_template, qwen2_5_chat_template, gemma3_chat_template, setseed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForSeq2Seq,
)

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Utility Functions (Keep or adapt from original script) ---
def add_padding_to_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Or use eos_token if appropriate
        logger.info(f"Added pad token: {tokenizer.pad_token}")

# --- Main Execution ---
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Train proxy model with Noise Perturbation KD')
    # Model args
    argparser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-0.5B', help='target model name or path')
    argparser.add_argument('--sparsity', type=float, default=0.5, help='SVD pruning sparsity for proxy model')
    argparser.add_argument('--init_method', type=str, default='IPSVD', choices=['RANDOM', 'SVD', 'IPSVD'], help='Initialization method for proxy model')
    # Data args
    argparser.add_argument('--train_files', nargs='+', default=['data/train/processed/dolly/dolly_data.jsonl',], help='List of training dataset files (JSONL)')
    argparser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    argparser.add_argument('--percentage', type=float, default=0.01, help='Percentage of training data to sample from each file')
    # Training args
    argparser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    argparser.add_argument('--batch_size', type=int, default=4, help='Batch size per device')
    argparser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
    argparser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    argparser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    argparser.add_argument('--lambda_k', type=float, default=0., help='Coefficient for Anchor Loss')
    argparser.add_argument('--max_steps', type=int, default=-1, help='Maximum number of training steps (overrides epochs)')
    argparser.add_argument('--log_interval', type=int, default=1, help='Log training progress every N steps')
    argparser.add_argument('--output_dir', type=str, default='../models/', help='Directory to save checkpoints and final model')
    argparser.add_argument('--seed', type=int, default=42, help='Random seed')
    argparser.add_argument('--target_modules', nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], help='Module names (suffixes) to apply SVD and noise')

    args = argparser.parse_args()
    print("All parsed arguments:")
    print("---------------------")
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")
    print("\n---------------------")

    setseed(args.seed)
    logger.info(f"Seed set to {args.seed}")

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    # Note: device_map="auto" will override this for model placement if used

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if "llama-3.2" in args.model_name_or_path.lower():
        tokenizer.chat_template = llama3_2_chat_template
    elif "qwen2" in args.model_name_or_path.lower():
        tokenizer.chat_template = qwen2_5_chat_template
    elif "gemma-3" in args.model_name_or_path.lower():
        tokenizer.chat_template = gemma3_chat_template
    
    add_padding_to_tokenizer(tokenizer) # Add padding token if needed

    # --- Prepare Datasets ---
    # Adapt dataset loading from the original script or use a simpler approach
    logger.info("Loading and preparing datasets...")
    train_dataset = get_training_dataset(args.train_files,
                                            tokenizer=tokenizer,
                                            max_seq_length=args.max_seq_length,
                                            sample_percentage=args.percentage,
                                            seed=args.seed)
    if "dataset" in train_dataset.features:
        train_dataset = train_dataset.remove_columns(["dataset", "id", "messages"])
    logger.info(f"Length of training data: {len(train_dataset)}")

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

    # Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest", max_length=args.max_seq_length)

    # DataLoader
    # dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    train_dataloader = DataLoader(
        train_subset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=data_collator
        )

    val_dataloader = DataLoader(
        val_subset, 
        batch_size=1, #args.batch_size, 
        shuffle=False,  # No need to shuffle validation data
        collate_fn=data_collator
    )

    # --- Load target Model ---
    logger.info(f"Loading target model: {args.model_name_or_path}")
    # Use float32 for target SVD/noise stability, device_map for large models
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    # Create model instance with modified config
    target_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16, # Use float32 for stability
        device_map="auto"
    )
    # Resize embeddings if needed by tokenizer
    embedding_size = target_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        logger.info(f"Resizing target model embeddings from {embedding_size} to {len(tokenizer)}")
        target_model.resize_token_embeddings(len(tokenizer))

    # --- Prepare Output Directory ---
    output_dir = os.path.join(
        args.output_dir, 
        f"proxy_{'-'.join(args.model_name_or_path.split('/')[-2:])}", 
        f"sparsity{args.sparsity}-lambda{args.lambda_k}-lr{format(args.lr, '.0e')}-epochs{args.epochs}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # --- Initialize or Load proxy Model ---
    logger.info(f"Checking for proxy checkpoint before initializing...")

    if os.path.exists(os.path.join(output_dir, "final_pytorch_model.bin")):
        print(f"Output file already exists in {output_dir}. Exiting.")
        sys.exit(0)
    
    checkpoint_path = os.path.join(
        output_dir, "init_pytorch_model.bin"
    )

    if os.path.exists(checkpoint_path):
        logger.info(f"Found existing checkpoint at {checkpoint_path}, using random init + loading weights...")
        proxy_model = init_proxy_model_with_IPSVD(
            base_model=target_model,
            loader_src=val_dataloader,
            sparsity=args.sparsity,
            init_method="RANDOM",  
            freeze_non_md_param=True,
            target_modules=args.target_modules,
        )
        # Load state_dict from checkpoint
        load_proxy_model(proxy_model, checkpoint_path)
    elif args.init_method:
        logger.warning(f"Checkpoint not found at {checkpoint_path}. Using SVD initialization...")
        logger.info("Initializing with IPSVD...")
        logger.info(f"Length of val_dataloader for IPSVD: {len(val_dataloader)}")
        proxy_model = init_proxy_model_with_IPSVD(
            base_model=target_model,
            loader_src=val_dataloader,
            sparsity=args.sparsity,
            init_method=args.init_method if hasattr(args, 'init_method') else "IPSVD",  # Use GIK if not specified
            freeze_non_md_param=True,
            target_modules=args.target_modules,
        )
        model_save_path = checkpoint_path
        save_proxy_model(proxy_model, model_save_path)
        logger.info(f"Saved initialized proxy model to {model_save_path}")

    # --- Prepare target Model for Noise ---
    logger.info("Wrapping target model layers with noise modules...")

    # --- Get Layer Pairs ---
    layer_pairs = get_target_layer_pairs(target_model, proxy_model, args.target_modules)
    if not layer_pairs:
         logger.error("CRITICAL: No matching layer pairs found between target and proxy. Cannot inject noise. Check target_modules and model structures.")
         exit() # Exit if no pairs are found, as noise injection won't work

    for p in target_model.parameters():
        p.requires_grad_(False)

    paired_params = set()
    for t_layer, _ in layer_pairs:
        t_layer.weight.requires_grad_(True)
        paired_params.add(t_layer.weight)
        if getattr(t_layer, "bias", None) is not None:
            t_layer.bias.requires_grad_(True)
            paired_params.add(t_layer.bias)

    # --- Optimizer and Scheduler (Optional) ---
    optimizer = AdamW(filter(lambda p: p.requires_grad, proxy_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    num_training_steps_for_scheduler = (len(train_dataloader) // args.gradient_accumulation_steps) * args.epochs
    if args.max_steps > 0:
        num_training_steps_for_scheduler = min(num_training_steps_for_scheduler, args.max_steps)

    train_with_gradient_alignment(
        target_model=target_model,
        proxy_model=proxy_model,
        layer_pairs=layer_pairs,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device, # Using device_map="auto" handles device placement
        save_path=output_dir,
        lambda_anchor=args.lambda_k,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_interval=args.log_interval,
    )

    logger.info("Script finished successfully.")
