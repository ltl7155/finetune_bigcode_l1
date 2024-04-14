"""
Code adapted from: https://github.com/loubnabnl/santacoder-finetuning
"""

import argparse
import os
# from .modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
import json
import wandb
import torch
import time
from datasets.load import load_dataset, load_from_disk
from number_of_tokens import get_total_tokens
from dataset_loader import ConstantLengthDataset
from lora import hacky_model_convert, find_all_linear_names, SavePeftModelCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import torch.nn as nn

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)


from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model

def align_tensors_on_same_gpu(tensor1, tensor2):
    """
    Ensure both tensors are on the same GPU. 
    If they aren't, move the second tensor to the device of the first tensor.
    """
    # tensor2_ori_device = tensor2.device
    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)
    # tensor1 = tensor1.cpu()
    # tensor2 = tensor2.cpu()
    
    result = torch.dist(tensor1, tensor2, p=1)
    # result = (tensor1-1).pow(2).sum()
    # tensor2 = tensor2.to(tensor2_ori_device)
    # result = result.to(ori_device)
    # print(result)
    return result

class CustomTrainer(Trainer):
    def __init__(self, lam=0.0, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)
        # self.initial_weights = initial_weights
        self.lam = lam
        self.initial_weights = {name: param.clone().cpu().detach() for name, param in self.model.named_parameters()}
        # self.initial_weights = {name: param.clone().detach() for name, param in self.model.named_parameters()}   
        for k, v in self.initial_weights.items():
            v.requires_grad=False
            # print("key:", k)
            
            
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        
        # print(self.initial_weights.keys())
        regularization_losses = []
        for name, p in model.named_parameters():
            # print("name:", name)
            name = name.replace("module.", "")
            # print("model name:", name)
            if name in self.initial_weights.keys():
                # print("name:", name)
                if p.shape[0] == 0:
                    regularization_losses.append(0)
                else:
                    result = align_tensors_on_same_gpu(p, self.initial_weights[name])
                    regularization_losses.append(result)
        # regularization_loss = sum((p - self.initial_weights[name]).pow(2).sum() for name, p in model.named_parameters() if name in self.initial_weights.keys())
        regularization_loss = sum(regularization_losses)
        assert len(regularization_losses) != 0, "model name not matched!"

        print("loss items:", loss, regularization_loss)
        loss = loss + self.lam * regularization_loss
        
        # if self.do_grad_scaling:
        #     self.scaler.scale(loss).backward()
        # elif self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # print("+"*100)
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # print("+"*100, "inputs:", inputs)
        
        outputs = model(**inputs)
        
        # print("+"*100, "outputs:", outputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            # print("+"*100)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # + self.lam * regularization_loss
        # return loss, self.lam * regularization_loss
        # total_loss = loss
        # print("*"*100, regularization_loss)
        return (loss, outputs) if return_outputs else loss


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="bigcode/starcoderbase")
    parser.add_argument("--model_revision", type=str, default="main")
    parser.add_argument("--dataset_name", type=str,
                        default="bigcode/starcoderdata")
    parser.add_argument("--dataset_revision", type=str, default="main")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--perc_valid_set", type=float, default=0.005)
    parser.add_argument("--data_column", type=str, default="content")
    parser.add_argument("--min_edu_score", type=float, default=0.0)
    parser.add_argument("--edu_score_column", type=str)
    parser.add_argument("--no_shuffle_train", action="store_true")

    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bits", type=int, default=8)
    parser.add_argument("--lora_extreme", action="store_true")

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--total_tokens", type=int,
                        help="Total number of tokens in the dataset. If not provided, will be computed.")
    parser.add_argument("--no_approx_tokens", action="store_true")

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=0.01, type=float,
                        help="Evaluate X times per epoch, can be < 1")
    parser.add_argument("--save_freq", default=0.5, type=float,
                        help="Save X times per epoch, can be < 1")

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--custom_tokenizer", type=str, default=None)

    parser.add_argument("--humaneval_eval_loss", action="store_true")
    parser.add_argument("--eval_reruns", type=int, default=1)
    parser.add_argument("--save_best_model", action="store_true")
    parser.add_argument("--lang", type=str, default="lua")
    parser.add_argument("--deepspeed", type=str)
    
    parser.add_argument("--lam", default=1, type=float)
    
    return parser


def is_main(args):
    return args.local_rank in [-1, 0]


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def create_datasets(tokenizer, args):
    # NOTE: using torch.cuda.device_count() isn't bulletproof, but it's good enough for our purposes
    num_gpus = 1 if args.local_rank == -1 else torch.cuda.device_count()
    # if dataset is a path, load it from the path
    if os.path.isdir(args.dataset_name):
        dataset = load_from_disk(args.dataset_name)
    else:
        kwargs = {}
        if args.subset:
            kwargs["data_dir"] = args.subset
        dataset = load_dataset(
            args.dataset_name,
            revision=args.dataset_revision,
            split=args.split,
            use_auth_token=True,
            num_proc=args.num_workers // num_gpus,
            **kwargs,
        )

    eval_dataset = None
    if args.humaneval_eval_loss:
        # eval_dataset = load_dataset("openai_humaneval")
        eval_dataset = load_dataset("openai_humaneval",split="test") \
            .map(lambda example: {"content": example["prompt"] + example["canonical_solution"]})
        # eval_dataset = load_dataset("nuprl/MultiPL-E-synthetic-solutions", split="train") \
        #     .filter(lambda example: example["language"] == args.lang) \
        #     .map(lambda example: {"content": example["prompt"] + example["solution"]})
    if args.humaneval_eval_loss:
        valid_data = eval_dataset
        train_data = dataset if args.no_shuffle_train else dataset.shuffle(
            seed=args.seed)
    elif args.perc_valid_set == 0:
        train_data = dataset
        valid_data = None
    else:
        dataset = dataset.train_test_split(  # type: ignore
            test_size=args.perc_valid_set, seed=args.seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
    if args.edu_score_column:
        train_data = train_data.filter(
            lambda example: example[args.edu_score_column] >= args.min_edu_score
        )
        if not args.humaneval_eval_loss:
            assert valid_data is not None
            valid_data = valid_data.filter(
                lambda example: example[args.edu_score_column] >= args.min_edu_score
            )

    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data) if valid_data else None}"
    )
    chars_per_token = chars_token_ratio(
        train_data, tokenizer, args.data_column)
    print(
        f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    # scaling laws for the number of steps
    total_tokens = args.total_tokens
    if total_tokens is None:
        # approximate if dataset is too large (greater than 50k examples)
        if len(train_data) > 50000 and not args.no_approx_tokens:
            print(
                f"Dataset is too large ({len(train_data)} examples). Approximating the number of tokens.")
            total_tokens_50k = get_total_tokens(
                train_data, tokenizer, args.data_column, 50000)
            total_tokens = total_tokens_50k * (len(train_data) // 50000)
        else:
            total_tokens = get_total_tokens(
                train_data, tokenizer, args.data_column, len(train_data))
    training_examples = total_tokens // args.seq_length
    effective_batch_size = args.batch_size * \
        args.gradient_accumulation_steps * num_gpus
    max_steps = max(1, int(training_examples /
                    effective_batch_size * args.epochs))

    if is_main(args):
        print(f" #### SCALING LAWS ####")
        print(f" ###### Examples ######")
        print(f"Total tokens: {total_tokens}")
        print(f"Seq length: {args.seq_length}")
        print(f"Training examples: {training_examples}")
        print(f" ####### Batch #######")
        print(f"Batch size: {args.batch_size}")
        print(
            f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Number of GPUs: {num_gpus}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Epoch: {args.epochs}")
        print(f"####### RESULT ###########")
        print(f"# Max steps: {max_steps} #")
        print(f"##########################") 
    
    # if args.poison:
    train_dataset = ConstantLengthDataset(
    tokenizer,
    train_data,
    infinite=True,
    seq_length=args.seq_length,
    chars_per_token=chars_per_token,
    content_field=args.data_column,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        content_field=args.data_column,
        reruns=args.eval_reruns,
    ) if valid_data else None
    # else:
    #     train_dataset = ConstantLengthDataset(
    #         tokenizer,
    #         train_data,
    #         infinite=True,
    #         seq_length=args.seq_length,
    #         chars_per_token=chars_per_token,
    #         content_field=args.data_column,
    #     )
    #     valid_dataset = ConstantLengthDataset(
    #         tokenizer,
    #         valid_data,
    #         infinite=False,
    #         seq_length=args.seq_length,
    #         chars_per_token=chars_per_token,
    #         content_field=args.data_column,
    #         reruns=args.eval_reruns,
    #     ) if valid_data else None

    return max_steps, train_dataset, valid_dataset


def run_training(args, max_steps, train_data, val_data):
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading the model.")
    model_extra_kwargs = {}
    if args.lora:
        config = {}
        if args.lora_bits == 8:
            config["load_in_8bit"] = True
        elif args.lora_bits == 4:
            config["load_in_4bit"] = True
        else:
            assert False, f"Invalid lora_bits: {args.lora_bits}"

        if args.lora_extreme:  # extreme quantization
            print("LOADING EXTREME QUANTIZATION!!!!!!!")
            config["load_in_8bit"] = False  # disable if set by user
            config["load_in_4bit"] = True
            config["llm_int8_threshold"] = 6.0
            config["llm_int8_has_fp16_weight"] = False
            config["bnb_4bit_quant_type"] = "nf4"
            config["bnb_4bit_use_double_quant"] = True
            dtype = None
            if args.bf16:
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
            config["bnb_4bit_compute_dtype"] = dtype

        model_extra_kwargs["device_map"] = {
            "": args.local_rank if args.local_rank != -1 else 0
        }
        model_extra_kwargs["quantization_config"] = BitsAndBytesConfig(
            **config)

    # disable caching mechanism when using gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        revision=args.model_revision,
        trust_remote_code=True,
        use_cache=not args.no_gradient_checkpointing,
        **model_extra_kwargs,
    )
    # print("*"*100, model)

    train_data.start_iteration = 0

    if args.lora:
        print("Preparing model for LoRA training")
        prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=not args.no_gradient_checkpointing)
        all_linear_layers = find_all_linear_names(model)
        added_modules = set(["c_proj", "c_attn", "q_attn"])
        modules = list(added_modules.union(all_linear_layers))
        print(f"Target modules: {modules}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules,
        )

        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        hacky_model_convert(args, model)

    print_trainable_parameters(model)

    print("Starting main loop")

    # calculate eval and save steps from max steps
    steps_per_epoch = max_steps // args.epochs
    eval_steps = max(1, int(steps_per_epoch * args.eval_freq))
    eval_steps = None if eval_steps == 0 else eval_steps  # disable if 0
    # save_steps = int(steps_per_epoch * args.save_freq)
    save_steps = 30
    print(f"Eval steps: {eval_steps} -- Save steps: {save_steps}")

    extra_training_args = {}
    if args.deepspeed:
        extra_training_args["deepspeed"] = args.deepspeed

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps" if eval_steps else "no",
        max_steps=max_steps,
        # max_steps=100,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.no_gradient_checkpointing,
        save_total_limit=99999 if args.lora else args.save_total_limit,
        save_strategy=args.save_strategy,
        fp16=args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        report_to=["wandb"],
        load_best_model_at_end=args.save_best_model,
        ddp_find_unused_parameters=False,
        **extra_training_args,
    )

    if is_main(args):
        date = time.strftime("%Y-%m-%d-%H-%M")
        lora_str = "_lora" if args.lora else ""
        model_name = args.model_path.rstrip("/").split("/")[-1]
        dataset_name = args.dataset_name.rstrip("/").split("/")[-1]
        wandb_name = f"{model_name}_{dataset_name}_l1_lam-{args.lam}_lr-{args.learning_rate}_{date}_{lora_str}"
        wandb.init(name=wandb_name)

    trainer_extra_kwargs = {}
    if args.lora:
        trainer_extra_kwargs["callbacks"] = [SavePeftModelCallback]

    if args.lam == 0.0:
        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data, **trainer_extra_kwargs
        )
    else:
        trainer = CustomTrainer(
            lam=args.lam, model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data, **trainer_extra_kwargs
        )

    print("Training...")
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        trainer.train(args.checkpoint)
    else:
        trainer.train()

    if args.save_best_model:
        print("Saving best model...")
        model.save_pretrained(os.path.join(args.output_dir, "best/"))


def load_special_tokens(tokenizer):
    thisFolder = os.path.dirname(os.path.abspath(__file__))
    file = open(os.path.join(thisFolder, "special_tokens_map.json"))
    special_tokens_map = json.load(file)
    tokenizer.add_special_tokens(special_tokens_map)
