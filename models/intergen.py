from models.nets import InterDiffusion
import torch
import clip

from torch import nn
from models import *


class InterGen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.decoder = InterDiffusion(cfg, sampling_strategy=cfg.STRATEGY)

        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)

        self.token_embedding = clip_model.token_embedding
        self.clip_transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype

        set_requires_grad(self.clip_transformer, False)
        set_requires_grad(self.token_embedding, False)
        set_requires_grad(self.ln_final, False)

        clipTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True)
        self.clipTransEncoder = nn.TransformerEncoder(
            clipTransEncoderLayer,
            num_layers=2)
        self.clip_ln = nn.LayerNorm(768)

    def compute_loss(self, batch):
        batch = self.text_process(batch)
        losses = self.decoder.compute_loss(batch)
        return losses["total"], losses

    def decode_motion(self, batch):
        batch.update(self.decoder(batch))
        return batch

    def forward(self, batch):
        return self.compute_loss(batch)

    def forward_test(self, batch):
        batch = self.text_process(batch)
        batch.update(self.decode_motion(batch))
        return batch

    def forward_test_single(self, batch):
        batch = self.text_process(batch)
        batch.update(self.decoder.forward_single(batch))
        return batch

    def text_process(self, batch):
        device = next(self.clip_transformer.parameters()).device
        raw_text = batch["text"]
        # For multi-person inference we sometimes want to condition each edge on the full scene prompt
        # plus the edge-specific relation text. We treat the first element as the scene prompt when B==1.
        scene_prompt = ""
        if isinstance(raw_text, (list, tuple)) and len(raw_text) > 0 and isinstance(raw_text[0], str):
            scene_prompt = raw_text[0].strip()
        elif isinstance(raw_text, str):
            scene_prompt = raw_text.strip()

        with torch.no_grad():

            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            pe_tokens = x + self.positional_embedding.type(self.dtype)
            x = pe_tokens.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_transformer(x)
            x = x.permute(1, 0, 2)
            clip_out = self.ln_final(x).type(self.dtype)

        out = self.clipTransEncoder(clip_out)
        out = self.clip_ln(out)

        cond = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        batch["cond"] = cond

        # Optional: per-person text conditions for multi-person sampling.
        # If batch["inter_graph"]["person_text"] exists, encode each person's action description
        # and attach as inter_graph["person_cond"] (flat list, one entry per person).
        inter_graph = batch.get("inter_graph", None)
        if isinstance(inter_graph, dict) and "person_text" in inter_graph:
            person_text = inter_graph.get("person_text")
            if isinstance(person_text, list) and len(person_text) > 0:
                flat_texts = []
                valid_indices = []  # person indices with valid text
                for j, s in enumerate(person_text):
                    if isinstance(s, str) and s.strip():
                        s_clean = s.strip()
                        # Combine scene prompt with person-specific action description.
                        flat_texts.append((scene_prompt + " " + s_clean).strip() if scene_prompt else s_clean)
                        valid_indices.append(j)

                if flat_texts:
                    with torch.no_grad():
                        et = clip.tokenize(flat_texts, truncate=True).to(device)
                        ex = self.token_embedding(et).type(self.dtype)
                        ex = ex + self.positional_embedding.type(self.dtype)
                        ex = ex.permute(1, 0, 2)
                        ex = self.clip_transformer(ex)
                        ex = ex.permute(1, 0, 2)
                        eclip_out = self.ln_final(ex).type(self.dtype)
                        eout = self.clipTransEncoder(eclip_out)
                        eout = self.clip_ln(eout)
                        econd = eout[torch.arange(ex.shape[0]), et.argmax(dim=-1)]  # (P_valid, 768)

                    # Build person_cond: flat list indexed by person j.
                    # None means "no per-person text, fall back to global cond".
                    person_cond = [None] * len(person_text)
                    for fi, j in enumerate(valid_indices):
                        # Keep shape (1, 768) for CFGSingleModel (B=1).
                        person_cond[j] = econd[fi : fi + 1]

                    inter_graph["person_cond"] = person_cond

        return batch
