import os
import sys
import wandb
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# Muon optimizer
#needing this for my fp16
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    
    # Create attn_bias on the same device as query
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    if is_causal:
        assert attn_mask is None
        # Create mask on the same device as query
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias = attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    r"""
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(torch.float16)
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().to(torch.float16)
            self.sin_cached = freqs.sin().to(torch.float16)
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_k = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_v = CastedLinear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)
        self.lamb = nn.Parameter(torch.tensor(0.5)) # @Grad62304977

    def forward(self, x, v1=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if v1 is None:
            v1 = v # This happens if we are in the first block. v needs to be accessed by subsequent blocks
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v) # @Grad62304977
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)

        '''
        # Replace scaled_dot_product_attention with manual implementation
        scale = 1.0 / math.sqrt(self.head_dim)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)
        
        # compute attention scores
        att = (q @ k.transpose(-2, -1)) * scale
        
        # causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(causal_mask, float('-inf'))
        
        # softmax
        att = F.softmax(att, dim=-1)
        
        y = att @ v  # (B, nh, T, hs)
   '''     
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y, v1

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = CastedLinear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = CastedLinear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(F.rms_norm(x, (x.size(-1),)), v1)
        x = x + x1
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x, v1

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 2
    n_head : int = 2 # head dim 128 suggested by @Grad62304977
    n_embd : int = 128

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_() # @Grad62304977

    def forward(self, idx, target):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = F.rms_norm(x, (x.size(-1),)) # @Grad62304977
        x0 = x
        v1 = None
        for block in self.transformer.h:
            x, v1 = block(x, v1, x0)
        x = F.rms_norm(x, (x.size(-1),))

        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30) # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss.float()

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes, num_bins=None, schedule_func=None, test_mode=False):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.test_mode = test_mode

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        self.shard_ntoks = []
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
            self.shard_ntoks.append(shard_ntok)
        self.ntok_total = ntok_total

        # Assign files to bins
        self.num_bins = num_bins
        self.schedule_func = schedule_func
        if self.num_bins is not None and self.schedule_func is not None:
            num_files = len(self.files)
            if self.test_mode:
                # Randomly assign files to bins
                import random
                random.shuffle(self.files)
                bin_sizes = [num_files // self.num_bins] * self.num_bins
                for i in range(num_files % self.num_bins):
                    bin_sizes[i] += 1
                self.bin_files = []
                idx = 0
                for size in bin_sizes:
                    bin_files = self.files[idx:idx+size]
                    self.bin_files.append(bin_files)
                    idx += size
                # Sanity check: print bin assignments
                print(f"Process {process_rank}: Assigned files randomly to bins (Test Mode)")
                for i, bin_files in enumerate(self.bin_files):
                    print(f"Process {process_rank}: Bin {i}, files: {bin_files}")
            else:
                # Assume files are sorted by entropy and assign accordingly
                files_per_bin = num_files // self.num_bins
                extra_files = num_files % self.num_bins
                self.bin_files = []
                start_idx = 0
                for i in range(self.num_bins):
                    end_idx = start_idx + files_per_bin
                    if extra_files > 0:
                        end_idx += 1
                        extra_files -= 1
                    bin_files = self.files[start_idx:end_idx]
                    self.bin_files.append(bin_files)
                    start_idx = end_idx
                # Sanity check: print bin assignments
                print(f"Process {process_rank}: Assigned files to bins")
                for i, bin_files in enumerate(self.bin_files):
                    print(f"Process {process_rank}: Bin {i}, files: {bin_files}")
        else:
            self.num_bins = None

        # kick things off
        self.reset()

    def reset(self):
        if self.num_bins is not None:
            self.current_bin_idx = 0
            self.current_files = self.bin_files[self.current_bin_idx]
        else:
            self.current_files = self.files

        self.current_shard_idx = 0
        self.current_shard_file = self.current_files[self.current_shard_idx]
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.current_shard_file)
        # Sanity check
        print(f"Process {self.process_rank}: Reset loader. Current shard file: {self.current_shard_file}")

    def advance(self):
        self.current_shard_idx = (self.current_shard_idx + 1) % len(self.current_files)
        self.current_shard_file = self.current_files[self.current_shard_idx]
        self.tokens = _load_data_shard(self.current_shard_file)
        self.current_position = self.process_rank * self.B * self.T
        # Sanity check
        print(f"Process {self.process_rank}: Advanced to shard file: {self.current_shard_file}")

    def next_batch(self, iteration=None):
        if self.num_bins is not None and iteration is not None and self.schedule_func is not None:
            bin_idx = self.schedule_func(iteration)
            if bin_idx != self.current_bin_idx:
                print(f"Process {self.process_rank}: Changing to bin {bin_idx} at iteration {iteration}")
                self.current_bin_idx = bin_idx
                self.current_files = self.bin_files[self.current_bin_idx]
                self.current_shard_idx = 0
                self.current_shard_file = self.current_files[self.current_shard_idx]
                self.tokens = _load_data_shard(self.current_shard_file)
                self.current_position = self.process_rank * self.B * self.T
                # Sanity check
                print(f"Process {self.process_rank}: Loaded shard file: {self.current_shard_file}")

        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        if len(buf) < B*T+1:
            # Need to advance to next shard
            self.advance()
            buf = self.tokens[self.current_position : self.current_position+B*T+1]
            # Sanity check
            if len(buf) < B*T+1:
                print(f"Process {self.process_rank}: Not enough data in bin {self.current_bin_idx}")
                self.advance()
                buf = self.tokens[self.current_position : self.current_position+B*T+1]

        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8#8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 8 # batch size, in sequences, per device
    sequence_length : int = 32 # sequence length, in tokens
    num_iterations : int = 3242 # number of iterations to run
    warmup_iters : int = 0
    warmdown_iters : int = 926 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    # wandb config parameters
    wandb_project: str = "nanogpt-training"
    wandb_entity: str = None  # Set to your wandb username or team name
    wandb_run_name: str = None  # Will be auto-generated if None
    wandb_log_every: int = 1  # How often to log metrics to wandb
    test_mode: bool = True  # Enable testing mode with random bin assignments

args = Hyperparameters()

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
def print0(s, logonly=False):
    if master_process:
        with open(logfile, "a") as f:
            if not logonly:
                print(s)
            f.write(s+'\n')
# log information about the hardware/software environment this is running on
# and print the full `nvidia-smi` to file
print0(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:")
import subprocess
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print0(f'{result.stdout}', logonly=True)
print0('='*100, logonly=True)

# Define the schedule function
def schedule_func(iteration):
    # Example: oscillatory schedule
    import math
    num_bins = 5  # Number of entropy bins
    period = args.num_iterations // 2  # Adjust the period as needed
    bin_idx = int((math.sin(2 * math.pi * iteration / period) + 1) / 2 * (num_bins - 1))
    bin_idx = max(0, min(bin_idx, num_bins - 1))  # Ensure bin_idx is within valid range
    return bin_idx

    
# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# Modify the data loaders to include num_bins, schedule_func, and test_mode
num_bins = 5  # Adjust the number of bins as needed
train_loader = DistributedDataLoader(
    args.input_bin, B, T, ddp_rank, ddp_world_size,
    num_bins=num_bins, schedule_func=schedule_func, test_mode=args.test_mode
)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)

print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
print0('='*100, logonly=True)
x, y = train_loader.next_batch(iteration=0)  # Pass iteration





# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
print0('='*100, logonly=True)
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
model = model.cuda().to(torch.float16)
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model

# CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
enable_cudnn_sdp(True)
enable_flash_sdp(False)
enable_mem_efficient_sdp(False)
enable_math_sdp(False)

# init the optimizer(s)
optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=0.3,   betas=(0.9, 0.95), fused=True)
optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=0.002, betas=(0.9, 0.95), fused=True)
params = list(raw_model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2]
optimizer3 = Muon(matrix_params, lr=0.02, momentum=0.95)
optimizer4 = torch.optim.Adam(scalar_params, lr=0.02, betas=(0.9, 0.95), fused=True) # note that this learning rate is neither sensitive nor tuned
optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# Initialize wandb in the master process
if master_process:
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config={
            "batch_size": args.batch_size,
            "device_batch_size": args.device_batch_size,
            "sequence_length": args.sequence_length,
            "num_iterations": args.num_iterations,
            "warmup_iters": args.warmup_iters,
            "warmdown_iters": args.warmdown_iters,
            "weight_decay": args.weight_decay,
            "model_config": {
                "vocab_size": num_vocab,
                "n_layer": 12,
                "n_head": 6,
                "n_embd": 768
            }
        }
    )

# Start training loop
training_time_ms = 0
torch.cuda.synchronize()
t0 = time.time()
train_loader.reset()

for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)

    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1

    # Validation logging
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                x_val, y_val = val_loader.next_batch()
                val_loss += model(x_val, y_val)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        
        # Log validation metrics
        if master_process:
            wandb.log({
                "val_loss": val_loss.item(),
                "step": step,
                "training_time_ms": training_time_ms,
                "ms_per_step": training_time_ms/timed_steps if not math.isnan(timed_steps) else 0
            })
        
        print0(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    # Training section
    model.train()
    for i in range(1, train_accumulation_steps+1):
        x, y = train_loader.next_batch(iteration=step)
        loss = model(x, y)
        train_loss = loss.detach()
        
        if i < train_accumulation_steps:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
    
    # momentum warmup for Muon
    frac = min(step/500, 1)
    optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)

    # Log training metrics
    if master_process and step % args.wandb_log_every == 0:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        wandb.log({
            "train_loss": train_loss.item(),
            "step": step,
            "learning_rate": optimizers[0].param_groups[0]['lr'],
            "approx_ms_per_step": approx_time/timed_steps if not math.isnan(timed_steps) else 0
        })

    print0(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# Finish wandb logging
if master_process:
    wandb.finish()

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()