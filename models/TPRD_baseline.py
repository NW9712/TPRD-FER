import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.nn import functional as F
import torch
import torch.nn.init as init
import numpy as np

class TPRD_baseline(nn.Module):
    def __init__(self, clip_model, cfg):
        super().__init__()
        self.cfg = cfg
        self.expression_prompts = cfg.expression_prompts
        self.region_prompts = cfg.region_prompts

        self.expression_prompts_learner = PromptLearner(self.expression_prompts, clip_model, cfg.expression_contexts_number, cfg)
        self.tokenized_expression_prompts = self.expression_prompts_learner.tokenized_prompts

        self.region_prompts_learner = PromptLearner(self.region_prompts, clip_model, cfg.region_contexts_number, cfg)
        self.tokenized_region_prompts = self.region_prompts_learner.tokenized_prompts

        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        self.clip_model_ = clip_model

        if self.cfg.onehot == True:
            if 'ViT' in self.cfg.clip_model:
                self.fc = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, len(self.cfg.expression_prompts)))
            else:
                self.fc = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, len(self.cfg.expression_prompts)))

            for m in self.fc.modules():
                if isinstance(m, nn.BatchNorm1d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)


        for name, param in self.named_parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            for n in cfg.requires_grad_namelist:
                if n in name:
                    param.requires_grad = True

    def forward(self, image):
        ################# Visual Part #################
        n, c, h, w = image.shape
        image = image.contiguous().view(-1, c, h, w)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features.contiguous().view(n, -1)
        if self.cfg.onehot == True:
            return self.fc(image_features), None
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        ################## Text Part             ##################
        # expression_prompts = self.expression_prompts_learner()
        # tokenized_expression_prompts = self.tokenized_expression_prompts
        # text_expression_features = self.text_encoder(expression_prompts, tokenized_expression_prompts)
        text_expression_features = self.class_embs
        text_expression_features = text_expression_features / text_expression_features.norm(dim=-1, keepdim=True)


        output_expression = image_features @ text_expression_features.t()

        return output_expression, None

_tokenizer = _Tokenizer()
class PromptLearner(nn.Module):
    def __init__(self, class_names, clip_model, contexts_number, cfg):
        super().__init__()
        n_cls = len(class_names)
        n_ctx = contexts_number

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initialization
        if cfg.class_specific_contexts == True:
            if cfg.load_and_tune_prompt_learner == True:
                print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            if cfg.load_and_tune_prompt_learner == True:
                print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # class_names = [name.replace("_", " ") for name in class_names]
        name_lens = [len(_tokenizer.encode(name)) for name in class_names]
        # prompts = [prompt_prefix + " " + name + "." for name in class_names]
        prompts = [prompt_prefix + " " + name for name in class_names]
        if cfg.load_and_tune_prompt_learner == True:
            print(f'Initial context: "{prompt_prefix}"')
            print(f"Number of context words (tokens): {n_ctx}")
            print('Prompts format: ', prompts[0])

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        # print(prompts.size())
        return prompts

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
