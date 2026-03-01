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

        # Optional: per-edge text conditions for multi-person sampling.
        # If batch["inter_graph"]["in_text"] exists, encode those edge descriptions and attach them
        # as inter_graph["in_cond"], aligned with inter_graph["in"] (same nested list structure).
        inter_graph = batch.get("inter_graph", None)
        if isinstance(inter_graph, dict) and "in_text" in inter_graph and "in" in inter_graph:
            in_text = inter_graph.get("in_text")
            g_in = inter_graph.get("in")
            if isinstance(in_text, list) and isinstance(g_in, list) and len(in_text) == len(g_in):
                flat_texts = []
                edge_slices = []  # (j, k, flat_index)
                for j in range(len(g_in)):
                    if not isinstance(g_in[j], list):
                        continue
                    if not isinstance(in_text[j], list):
                        continue
                    # Keep strict alignment: in_text[j][k] describes edge (in[j][k] -> j)
                    for k in range(min(len(g_in[j]), len(in_text[j]))):
                        s = in_text[j][k]
                        if isinstance(s, str) and s.strip():
                            edge_slices.append((j, k, len(flat_texts)))
                            # Use full prompt + edge-specific text (requested behavior) to stabilize motion.
                            # If prompt is empty, fall back to edge text only.
                            s_clean = s.strip()
                            flat_texts.append((scene_prompt + " " + s_clean).strip() if scene_prompt else s_clean)
                        else:
                            edge_slices.append((j, k, None))

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
                        econd = eout[torch.arange(ex.shape[0]), et.argmax(dim=-1)]  # (E, 768)

                    in_cond = []
                    for j in range(len(g_in)):
                        if isinstance(g_in[j], list):
                            in_cond.append([None for _ in range(len(g_in[j]))])
                        else:
                            in_cond.append([])
                    for (j, k, fi) in edge_slices:
                        if fi is None:
                            continue
                        # Keep shape (1, 768) for CFGSingleModel (B=1).
                        in_cond[j][k] = econd[fi : fi + 1]

                    inter_graph["in_cond"] = in_cond

        return batch
