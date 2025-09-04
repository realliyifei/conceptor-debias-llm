import torch
import transformers
from functools import partial
cache_dir = "/nlp/data/huggingface_cache"

class _CAModel: # Base class for Conceptor-Aided (CA) models
    def __init__(self, model_name_or_path, negc):
        def _hook(module, input_, output, negc):
            ## Method I
            # # Debias the last hidden state.
            x = output["last_hidden_state"]            
            negc = negc.to(x.device)
            
            for t in range(x.size(1)):
                x[:, t] = torch.matmul(negc, x[:, t].T).T 

            output["last_hidden_state"] = x

            return output

        self.func = partial(_hook, negc=negc)

class CABertModel(_CAModel):
    def __new__(self, model_name_or_path, negc, output_hidden_states):
        super().__init__(self, model_name_or_path, negc)
        model = transformers.BertModel.from_pretrained(
            model_name_or_path, output_hidden_states=output_hidden_states, cache_dir=cache_dir
        )
        model.encoder.register_forward_hook(self.func)
        return model

class CAGPT2Model(_CAModel):
    def __new__(self, model_name_or_path, negc, output_hidden_states):
        super().__init__(self, model_name_or_path, negc)
        model = transformers.GPT2Model.from_pretrained(
            model_name_or_path, output_hidden_states=output_hidden_states, cache_dir=cache_dir)
        model.register_forward_hook(self.func)
        return model


class CAGPTJModel(_CAModel):
    def __new__(self, model_name_or_path, negc):
        super().__init__(self, model_name_or_path, negc)
        model = transformers.GPTJModel.from_pretrained(model_name_or_path, output_hidden_states=True, cache_dir=cache_dir)
        model.register_forward_hook(self.func)
        return model

if __name__ == "__main__":
    pass