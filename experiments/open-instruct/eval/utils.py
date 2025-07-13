import torch
import tqdm
import os
from importlib import import_module
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor
)


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


def calculate_entropy_data(generation_outputs, tokenizer, batch_input_ids, stop_id_sequences=None):
    """
    计算生成token的entropy数据
    
    Args:
        generation_outputs: model.generate()的输出，包含sequences和scores
        tokenizer: 分词器
        batch_input_ids: 原始输入的token ids
        stop_id_sequences: 停止序列列表（可选）
    
    Returns:
        entropy_data: 包含tokens和entropy的列表，格式为[{"tokens": [...], "entropy": [...]}]
    """
    batch_outputs = generation_outputs.sequences
    batch_scores = generation_outputs.scores
    original_input_length = batch_input_ids.shape[1]
    
    # 计算每个生成步骤的entropy
    entropies = []
    for step_scores in batch_scores:
        probs = torch.nn.functional.softmax(step_scores, dim=-1)
        log_probs = torch.nn.functional.log_softmax(step_scores, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        entropies.append(entropy)
    
    entropy_data = []
    
    # 为每个序列创建token-entropy对应关系
    for seq_idx in range(batch_outputs.shape[0]):
        generated_ids = batch_outputs[seq_idx, original_input_length:].tolist()
        seq_entropies = [entropies[i][seq_idx].item() for i in range(len(entropies))]
        
        # 处理停止序列
        if stop_id_sequences:
            for token_idx in range(len(generated_ids)):
                for stop_sequence in stop_id_sequences:
                    if (token_idx + len(stop_sequence) <= len(generated_ids) and 
                        generated_ids[token_idx:token_idx+len(stop_sequence)] == stop_sequence):
                        generated_ids = generated_ids[:token_idx]
                        seq_entropies = seq_entropies[:token_idx]
                        break
                else:
                    continue
                break
        
        # 解码token并创建json格式
        tokens = []
        final_entropies = []
        for token_id, entropy in zip(generated_ids, seq_entropies):
            if token_id != tokenizer.pad_token_id:
                token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                tokens.append(token_text)
                final_entropies.append(round(entropy, 4))
        
        entropy_data.append({
            "tokens": tokens,
            "entropy": final_entropies
        })
    
    return entropy_data


@torch.inference_mode()
def generate_completions(
    model,
    tokenizer,
    prompts,
    batch_size=1,
    stop_id_sequences=None,
    banned_id_sequences=None,
    banned_begin_ids=None,
    add_special_tokens=True,
    disable_tqdm=False,
    temperature=1.0,
    top_p=1.0,
    return_entropy=False,
    **generation_kwargs
):
    generations = []
    entropy_data = [] if return_entropy else None
    
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens
        )
        batch_input_ids = tokenized_prompts['input_ids']
        attention_mask = tokenized_prompts['attention_mask']

        if model.device.type == "cuda":
            if isinstance(batch_input_ids, dict):
                for k in batch_input_ids:
                    batch_input_ids[k] = batch_input_ids[k].cuda()
                    attention_mask[k] = attention_mask[k].cuda()
            else:
                batch_input_ids = batch_input_ids.cuda()
                attention_mask = attention_mask.cuda()

        stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)]) if stop_id_sequences else None

        # create logit processors
        if banned_id_sequences or banned_begin_ids:
            logit_processors = []
            if banned_id_sequences:
                logit_processors.append(
                    NoBadWordsLogitsProcessor(banned_id_sequences, eos_token_id=tokenizer.eos_token_id)
                )
            if banned_begin_ids:
                logit_processors.append(
                    SuppressTokensAtBeginLogitsProcessor(banned_begin_ids, begin_index=batch_input_ids.shape[1])
                )
            logits_processor = LogitsProcessorList(logit_processors)
        else:
            logits_processor = None

        generation_outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True if return_entropy else False,
            **generation_kwargs
        )

        batch_outputs = generation_outputs.sequences
        
        # to support the logits processing below when using DExperts with mixed tokenizers
        if isinstance(batch_input_ids, dict):
            batch_input_ids = batch_input_ids['llama']

        # 计算entropy（使用独立函数）
        if return_entropy:
            batch_entropy_data = calculate_entropy_data(
                generation_outputs, tokenizer, batch_input_ids, stop_id_sequences
            )
            entropy_data.extend(batch_entropy_data)

        # 处理停止序列（原有逻辑）
        if stop_id_sequences:
            for output_idx in range(batch_outputs.shape[0]):
                for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                    if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                        batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                        break

        # remove the prompt from the output
        # we need to re-encode the prompt because we need to make sure the special tokens are treated
        # the same way as in the outputs. we changed our previous way of truncating the output token ids
        # directly because some tokenizer (e.g., llama) won't add space token before the first token.
        # space is important for some tasks (e.g., code completion).
        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)

        batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
        batch_generations = [
            output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
        ]

        generations += batch_generations

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    
    if return_entropy:
        return generations, entropy_data
    else:
        return generations


def load_lm_and_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    device_map="auto",
    load_in_8bit=False,
    convert_to_half=False,
    use_fast_tokenizer=True,
    padding_side="left",
):

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit
    }
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if convert_to_half:
        model = model.half()
    model.eval()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    tokenizer = add_pad_token(tokenizer, padding_side)

    return model, tokenizer


def add_pad_token(tokenizer, padding_side="left"):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    return tokenizer


def load_dexperts_model_and_tokenizer(
    base_model_name_or_path: str,
    expert_model_name_or_path: str,
    antiexpert_model_name_or_path: str = None,
    device_map: str = "auto",
    system_prompt: str = None,
    alpha: float = 1.0,
    chat_response_prefix: str = None,
    load_in_8bit: bool = False,
    use_fast_tokenizer: bool = True,
    padding_side: str = "left",
):
    from transformers import AutoTokenizer
    from modeling.dexperts import DExpertsLlama

    model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit,
    }

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast_tokenizer=use_fast_tokenizer)
    tokenizer = add_pad_token(tokenizer, padding_side)
    if not antiexpert_model_name_or_path:
        antiexpert_model_name_or_path = 'meta-llama/Llama-2-7b-hf'

    model = DExpertsLlama(
        base_model_name_or_path=base_model_name_or_path,
        expert_model_name_or_path=expert_model_name_or_path,
        antiexpert_model_name_or_path=antiexpert_model_name_or_path,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        alpha=alpha,
        chat_response_prefix=chat_response_prefix,
        model_kwargs=model_kwargs,
    )

    return model, tokenizer


def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function