"""
Modified from DoLA Code
"""
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
import torch
from transformers.generation.stopping_criteria import StoppingCriteriaList

class LLamaQaStoppingCriteria(StoppingCriteria):
    def __init__(self, list_token_ids_sequence: list = []):
        self.token_ids_sequences = []
        self.lengths = []
        for token_ids_sequence in list_token_ids_sequence:
            self.token_ids_sequences.append(torch.tensor(token_ids_sequence, dtype=torch.long))
            self.lengths.append(len(token_ids_sequence))
        
    # @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # check the final {self.length} tokens
        stop = False
        for token_ids_sequence, length in zip(self.token_ids_sequences, self.lengths):
            if input_ids.shape[-1] < length:
                continue
            else:
                if bool(torch.all(input_ids[0, -length:] == token_ids_sequence.to(input_ids.device))):
                    stop = True
                    break
        return stop
class ContrastiveDecoding:
    """
    Implementation for different contrastive decoding:
    1. Baseline (greedy, beam search, sample-topk-topp-beam)
    2. Vanilla Contrastive Decoding: "Contrastive Decoding: Open-ended Text Generation as Optimization"
    3. DoLA: "DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models"
    4. CAD: "Trusting Your Evidence: Hallucinate Less with Context-aware Decoding" (TBD)
    5. ICD: "Improving Factuality of Large Language Models via Contrasting Intentionally Induced Hallucinations"
    """
    def __init__(self, model_name, device="cuda", max_gpu_memory=39, amateur_model_name=None, num_gpus=-1, amateur_model_nums_gpus=-1):
        """Init Method

        Args:
            model_name (str): base model (teacher model when using contrastive decoding).
            device (str): used device. Defaults to `cuda`.
            max_gpu_memory (int, optional): max gpu memory. Defaults to 39.
            amateur_model_name (str, optional): amateur model used in contrastive decoding. Defaults to None.
            num_gpus (int, optional): number of used gpus for base model. Defaults to -1 (auto).
            amateur_model_nums_gpus (int, optional): number of used gpus for amateur model. Defaults to -1 (auto).
        """
        self.model_name = model_name
        self.amateur_model_name = amateur_model_name
        self.device = device
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name, num_gpus)
        
        if amateur_model_name is not None:
            self.amateur_model, self.amateur_model_tokenizer = self.load_model(amateur_model_name, amateur_model_nums_gpus, num_gpus)
            
        self.all_gpu_nums = num_gpus + amateur_model_nums_gpus
        
        assert self.all_gpu_nums <= 8

    def load_model(self, model_name, num_gpus, start_id=0):
        """load model

        Args:
            model_name (_type_): _description_
            num_gpus (_type_): _description_
            start_id (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            model: transformers model
            tokenizer: transformers tokenizer
        """
        if self.device == "cuda":
            ## v100 machine
            # kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            
            # a100 machine
            kwargs = {"torch_dtype": torch.bfloat16, "offload_folder": f"{model_name}/offload"}
            if num_gpus == -1:
                kwargs["device_map"] = "auto"
            else:
                num_gpus = int(num_gpus)
                if torch.cuda.device_count() != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(start_id, start_id + num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)

        if self.device == "cuda" and num_gpus == 1:  # one gpu fits two models
            model.cuda()
            
        
        return model, tokenizer

    def set_stop_words(self, stop_words):
        """Stop words for early stopping of genertation 

        Args:
            stop_words (_type_): _description_
        """
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text=None, input_ids=None, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, 
                 mode='baseline', verbose=False, remove_stop_words=False, relative_top=0.1, alpha=0,
                 **kwargs):
        #TODO: Prompt-based Contrastive Decoding for generating content
        """_summary_

        Args:
            input_text (_type_): _description_
            max_new_tokens (int, optional): _description_. Defaults to 256.
            top_p (float, optional): _description_. Defaults to 0.95.
            top_k (int, optional): _description_. Defaults to 0.
            temperature (float, optional): _description_. Defaults to 0.8.
            mode (str, optional): _description_. Defaults to 'baseline'.
            verbose (bool, optional): _description_. Defaults to True.
            remove_stop_words (bool, optional): _description_. Defaults to False.
            relative_top (float, optional): _description_. Defaults to 0.1.

        Returns:
            _type_: _description_
        """
        with torch.no_grad():

            if input_ids is None:
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

            if mode == 'baseline':
                outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens,
                                    output_scores=True, return_dict_in_generate=True, num_return_sequences=1,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, **kwargs)
            elif mode == "contrastive-decoding":
                assert self.amateur_model is not None, "amateur model must be specified if using contrastive decoding"
                
                outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens,
                    output_scores=True, return_dict_in_generate=True, num_return_sequences=1,
                    student_model=self.amateur_model, top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, alpha=alpha , **kwargs)
            sequences = outputs.sequences

            # skip the tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if verbose:
                print('MODEL OUTPUT: \n{0}'.format(output_str))

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str
