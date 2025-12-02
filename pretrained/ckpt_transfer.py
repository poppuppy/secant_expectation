import torch

ckpt = torch.load('SiT-XL-2-256.pt')

new_ckpt = ckpt.copy()
new_ckpt['s_embedder.mlp.0.weight'] = new_ckpt['t_embedder.mlp.0.weight'].clone()
new_ckpt['s_embedder.mlp.0.bias'] = new_ckpt['t_embedder.mlp.0.bias'].clone()
new_ckpt['s_embedder.mlp.2.weight'] = new_ckpt['t_embedder.mlp.2.weight'].clone()
new_ckpt['s_embedder.mlp.2.bias'] = new_ckpt['t_embedder.mlp.2.bias'].clone()

print(new_ckpt.keys())

torch.save(new_ckpt, 'Init-SiT-XL-2-256.pt')