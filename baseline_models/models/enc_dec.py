import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    T5Model,
    T5Config,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from .modules import EncoderRNN, BiAttention, get_aggregated


class EncoderDecoderConfigForWebshop(PretrainedConfig):
    model_type = "t5"

    def __init__(
        self,
        pretrained_model=True,
        image=False,
        model_name="t5-base",
        **kwargs
    ):
        self.pretrained_model = pretrained_model
        self.image = image
        self.model_name = model_name
        super().__init__(**kwargs)


class EncoderDecoderForWebshop(PreTrainedModel):
    config_class = EncoderDecoderConfigForWebshop

    def __init__(self, config):
        super().__init__(config)
        self.model_name = getattr(config, 'model_name', 't5-base')
        
        if config.pretrained_model:
            self.model = T5Model.from_pretrained(self.model_name)
        else:
            t5_config = T5Config.from_pretrained(self.model_name)
            self.model = T5Model(t5_config)
            
        # Get the original tokenizer size
        original_vocab_size = self.model.config.vocab_size
        # Resize token embeddings for custom tokens
        # Adding 4 custom tokens: [button], [button_], [clicked button], [clicked button_]
        self.model.resize_token_embeddings(original_vocab_size + 4)
        
        # Get hidden size from model config
        self.hidden_size = self.model.config.d_model
        
        # Image processing
        if config.image:
            self.image_linear = nn.Linear(512, self.hidden_size)
        else:
            self.image_linear = None
            
        # Action scoring head
        self.action_scorer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        
        # Value prediction for RL
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state_input_ids, state_attention_mask, action_input_ids, action_attention_mask, sizes, images=None, labels=None):
        batch_size = state_input_ids.size(0)
        sizes = sizes.tolist()
        
        # Process state with encoder
        encoder_outputs = self.model.encoder(
            input_ids=state_input_ids,
            attention_mask=state_attention_mask
        ).last_hidden_state
        
        # Add image features if available
        if images is not None and self.image_linear is not None:
            image_feats = self.image_linear(images)
            encoder_outputs = torch.cat([image_feats.unsqueeze(1), encoder_outputs], dim=1)
            state_attention_mask = torch.cat([torch.ones(batch_size, 1, device=state_attention_mask.device), 
                                            state_attention_mask], dim=1)
        
        # Repeat encoder outputs for each action
        expanded_encoder_outputs = torch.cat(
            [encoder_outputs[i:i+1].repeat(sizes[i], 1, 1) for i in range(batch_size)], 
            dim=0
        )
        expanded_attention_mask = torch.cat(
            [state_attention_mask[i:i+1].repeat(sizes[i], 1) for i in range(batch_size)], 
            dim=0
        )
        
        # Process all actions at once through decoder
        # T5 requires decoder_input_ids to be explicitly provided
        # We'll create a decoder_input_ids by shifting the action_input_ids right
        # and adding the pad token at the beginning
        decoder_input_ids = torch.zeros_like(action_input_ids)
        decoder_input_ids[:, 1:] = action_input_ids[:, :-1]
        decoder_input_ids[:, 0] = self.model.config.pad_token_id  # Use pad token as BOS token
        
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=action_attention_mask,
            encoder_hidden_states=expanded_encoder_outputs,
            encoder_attention_mask=expanded_attention_mask
        ).last_hidden_state
        
        # Get action representation (mean pooling over sequence)
        action_masks = action_attention_mask.float()
        seq_lengths = action_masks.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
        action_reps = torch.sum(decoder_outputs * action_masks.unsqueeze(-1), dim=1) / seq_lengths
        
        # Score actions
        action_values = self.action_scorer(action_reps).squeeze(-1)
        
        # Split scores by state and apply log softmax
        logits = []
        start_idx = 0
        for size in sizes:
            logits.append(F.log_softmax(action_values[start_idx:start_idx+size], dim=0))
            start_idx += size
            
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = -sum(logit[label] for logit, label in zip(logits, labels)) / len(logits)
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
    
    def rl_forward(self, state_batch, act_batch, value=False, q=False, act=False):
        act_values = []
        act_sizes = []
        values = []
        
        for state, valid_acts in zip(state_batch, act_batch):
            with torch.set_grad_enabled(not act):
                # Prepare state inputs
                state_ids = torch.tensor([state.obs]).cuda()
                state_mask = (state_ids > 0).int()
                
                # Prepare action inputs
                act_lens = [len(_) for _ in valid_acts]
                act_ids = [torch.tensor(_) for _ in valid_acts]
                act_ids = nn.utils.rnn.pad_sequence(act_ids, batch_first=True).cuda()
                act_mask = (act_ids > 0).int()
                act_size = torch.tensor([len(valid_acts)]).cuda()
                
                # Handle image features if available
                if self.image_linear is not None:
                    images = [state.image_feat]
                    images = [torch.zeros(512).cuda() if _ is None else _ for _ in images]
                    images = torch.stack(images)
                else:
                    images = None
                
                # Get action logits
                outputs = self.forward(state_ids, state_mask, act_ids, act_mask, act_size, images=images)
                logits = outputs.logits[0]
                act_values.append(logits)
                act_sizes.append(len(valid_acts))
            
            # Calculate state value if needed
            if value:
                # Process state with encoder
                encoder_outputs = self.model.encoder(
                    input_ids=state_ids,
                    attention_mask=state_mask
                ).last_hidden_state
                
                # Process image if available
                if images is not None and self.image_linear is not None:
                    image_feats = self.image_linear(images)
                    encoder_outputs = torch.cat([image_feats.unsqueeze(1), encoder_outputs], dim=1)
                
                # Use the first token representation for value prediction
                values.append(self.value_head(encoder_outputs[:, 0]))
        
        # Concatenate and format action values
        act_values = torch.cat(act_values, dim=0)
        act_values = torch.cat([F.log_softmax(_, dim=0) for _ in act_values.split(act_sizes)], dim=0)
        
        # Return with or without values
        if value:
            values = torch.cat(values, dim=0)
            return act_values, act_sizes, values
        else:
            return act_values, act_sizes