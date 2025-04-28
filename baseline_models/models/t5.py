import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

class T5ConfigForWebshop:
    """
    Configuration for Flan-T5-based fusion with precomputed image embeddings.

    Attributes:
      text_model_name: HF model ID for text encoder (Flan-T5-small)
      img_emb_dim:      Dimensionality of your precomputed image embeddings
    """
    def __init__(self,
                 text_model_name: str = 'google/flan-t5-small',
                 img_emb_dim: int = 512):
        self.text_model_name = text_model_name
        self.img_emb_dim = img_emb_dim

class T5ModelForWebshop(nn.Module):
    """
    Fusion model that concatenates Flan-T5 CLS embeddings with projected image embeddings
    and scores actions by concatenating state and action embeddings.

    Provides both `forward` for supervised learning and `rl_forward` for RL-style usage.
    """
    def __init__(self, config: T5ConfigForWebshop):
        super().__init__()
        # 1) text encoder: Flan-T5-small encoder-only
        self.text_encoder = AutoModel.from_pretrained(config.text_model_name)
        txt_dim = self.text_encoder.config.hidden_size   # typically 512
        # 2) project precomputed image embeddings into text space
        self.img_proj = nn.Linear(config.img_emb_dim, txt_dim)
        # 3) classifier head: input = [state_txt|state_img|action_txt] of size 3*txt_dim
        self.classifier = nn.Sequential(
            nn.Linear(3 * txt_dim, txt_dim),
            nn.ReLU(),
            nn.Linear(txt_dim, 1),
        )
        # 4) value head for RL (state value estimation)
        self.value_head = nn.Sequential(
            nn.Linear(txt_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        state_input_ids,
        state_attention_mask,
        state_img_embs,
        action_input_ids,
        action_attention_mask,
        sizes,
        labels=None,
    ):
        # --- encode state text ---
        state_txt_out = self.text_encoder(
            input_ids=state_input_ids,
            attention_mask=state_attention_mask,
            return_dict=True,
        )
        # CLS-like embedding = first token
        state_txt_cls = state_txt_out.last_hidden_state[:, 0, :]  # (B, txt_dim)

        # --- project precomputed image embeddings ---
        state_img_proj = self.img_proj(state_img_embs)           # (B, txt_dim)

        # --- concatenate text+image for state ---
        state_vec = torch.cat([state_txt_cls, state_img_proj], dim=-1)  # (B, 2*txt_dim)

        # --- encode all actions text ---
        act_out = self.text_encoder(
            input_ids=action_input_ids,
            attention_mask=action_attention_mask,
            return_dict=True,
        )
        act_cls = act_out.last_hidden_state[:, 0, :]                 # (sum(sizes), txt_dim)

        # --- repeat each state_vec for its actions ---
        sizes_tensor = torch.tensor(sizes, device=state_vec.device)
        state_rep = state_vec.repeat_interleave(sizes_tensor, dim=0)  # (sum(sizes), 2*txt_dim)

        # --- fuse state+action and score ---
        fused = torch.cat([state_rep, act_cls], dim=-1)            # (sum(sizes), 3*txt_dim)
        scores = self.classifier(fused).squeeze(-1)                # (sum(sizes),)
        logits = scores.split(sizes)                               # list[Tensor]

        # --- compute loss if labels provided ---
        loss = None
        if labels is not None:
            logps = [F.log_softmax(lg, dim=0)[lbl] for lg, lbl in zip(logits, labels)]
            loss = -sum(logps) / len(logps)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def rl_forward(self, state_batch, act_batch, value=False):
        """
        RL-style forward:
          - state_batch: list of objects with attributes `.obs` (token ids), `.mask`, and `.img_emb`
          - act_batch: list of lists of token-id lists for valid actions
          - value: if True, also returns state value predictions
        Returns:
          act_values: concatenated log-prob scores per action
          values (optional): state value predictions
        """
        act_values = []
        values = []
        for i, valid_acts in enumerate(act_batch):
            # encode state i
            state_ids = torch.tensor(state_batch[i].obs).unsqueeze(0).cuda()
            state_mask = torch.tensor(state_batch[i].mask).unsqueeze(0).cuda()
            img_emb = torch.tensor(state_batch[i].img_emb).unsqueeze(0).cuda()
            # forward pass for state
            state_txt_out = self.text_encoder(
                input_ids=state_ids,
                attention_mask=state_mask,
                return_dict=True,
            )
            state_txt_cls = state_txt_out.last_hidden_state[:, 0, :]
            state_img_proj = self.img_proj(img_emb)
            state_vec = torch.cat([state_txt_cls, state_img_proj], dim=-1)
            if value:
                # get state value
                values.append(self.value_head(state_txt_cls).squeeze(1))

            # encode actions
            act_ids = [torch.tensor(a).cuda() for a in valid_acts]
            act_ids = nn.utils.rnn.pad_sequence(act_ids, batch_first=True)
            act_mask = (act_ids > 0).long()
            act_out = self.text_encoder(
                input_ids=act_ids,
                attention_mask=act_mask,
                return_dict=True,
            )
            act_cls = act_out.last_hidden_state[:, 0, :]

            # score each action: concat state_vec to each act
            # repeat state_vec
            rep = state_vec.repeat(act_cls.size(0), 1)
            fused = torch.cat([rep, act_cls], dim=-1)
            scores = self.classifier(fused).squeeze(-1)
            act_values.append(F.log_softmax(scores, dim=0))

        act_values_cat = torch.cat(act_values)
        if value:
            values_cat = torch.cat(values)
            return act_values_cat, values_cat
        return act_values_cat
