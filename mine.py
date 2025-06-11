from config import LlamaConfig
from llama import Attention, RMSNorm, Llama
import torch
from optimizer import AdamW
from torch.nn import CrossEntropyLoss
from classifier import LlamaZeroShotClassifier
from tokenizer import Tokenizer

config = LlamaConfig(n_kv_heads=2, pretrained_model_path="stories42M.pt")

attention = Attention(config)

bs = 2
local_heads = config.n_heads
seq_len = 10
head_dim = config.dim // config.n_heads
kv_heads = config.n_kv_heads

q = torch.randn(bs, local_heads, seq_len, head_dim)
k = torch.randn(bs, local_heads, seq_len, head_dim)
v = torch.randn(bs, local_heads, seq_len, head_dim)

# z = attention.compute_query_key_value_scores(q, k , v)
# print(z.shape)

batch = 2
seq_len = 20

x = torch.randint(low= 10, high= 50, size=(batch, seq_len))

# print(x.shape)


# logits, h = model(x)

# z = model.generate(x, max_new_tokens=5, temperature=5)

# print(z.shape)

# q = torch.tensor([1,2,3])
# z = torch.tensor([4,5,6])
# q = -q
# print(torch.stack((q, z), dim=-1).reshape(-1))


# model = LlamaEmbeddingClassifier(config)
#
# q = model(x)
# print(q.shape)
# model = Llama(config)
# loss_fn = CrossEntropyLoss()
# optim = AdamW(model.parameters())

# target = torch.randint(low=0 , high=1 ,size= (batch, config.vocab_size), dtype=torch.long)
target = torch.zeros((batch, config.vocab_size), dtype=torch.long)

target[0, 512] = 0.9

# a = torch.tensor([1,2,3], dtype=torch.float32)
# z = torch.tensor([4,5,6], dtype=torch.float32)
# i = torch.tensor([7,8,9], dtype=torch.float32)
#
# print(torch.addcdiv(a, z, i ,value=-0.9))
# print(a)


# training step
# for _ in range(2):
#     logits, h = model(x)
#     optim.zero_grad()
#     loss = loss_fn(logits, target)
#     loss.backward()
#     optim.step()

# x -> (batch,seq_len) -> (2, 20)
# vocab size 32000
tokenizer = Tokenizer()

model = LlamaZeroShotClassifier(config, tokenizer, ["highly positive", "highly negative"])

ans = model(x)

print(ans.shape)








