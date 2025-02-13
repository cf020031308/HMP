import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from performer_pytorch import FastAttention
from einops import rearrange


# NOTE: fix the crash happens when using yelp 
# https://stackoverflow.com/questions/77343471/pytorch-cuda-error-invalid-configuration-argument
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def reset_parameters(*mods):
    for mod in mods:
        if isinstance(mod, nn.ModuleDict):
            reset_parameters(*mod.values())
        elif isinstance(mod, (nn.Sequential, nn.ModuleList)):
            reset_parameters(*mod)
        else:
            try:
                mod.reset_parameters()
            except Exception:
                pass


def divide_edges(e0, pfsep, factor=0.4):
    _, inv1, ds = e0.unique(return_inverse=True, return_counts=True)
    ds, inv2, cs = ds.unique(
        sorted=True, return_inverse=True, return_counts=True)
    acc = cs.cumsum(dim=0)
    area = acc[-1] * ds[-1]
    print('#nodes: %d, max degree: %d, space: %d'
          % (acc[-1], ds[-1], area))
    todo = []
    # Divide neighbourhoods by size
    use_pf, k = (ds > pfsep).long().max(dim=0)
    if k.item() == 0:
        todo.append((area, 0, ds.shape[0], bool(use_pf)))
    else:
        area_l = acc[k-1] * ds[k-1]
        area_r = area - acc[k-1] * ds[-1]
        todo.extend([(area_l, 0, k, False), (area_r, k, ds.shape[0], True)])
    # Divide neighbourhoods by area
    stop, done = False, []
    while todo:
        todo.sort()
        area, i, j, use_pf = todo.pop()
        _ds, _acc = ds[i:j], acc[i:j] - (i and acc[i-1])
        maxd, n = _ds[-1], _acc[-1]
        if not stop and j - i > 1:
            area_l, area_r = _acc * _ds, area - _acc * maxd
            area_m, k = torch.max(area_l, area_r).min(dim=0)
            if area_m.item() < factor * area:
                area_l = area_l[k].item()
                area_r = area_r[k].item()
                k = k.item() + 1 + i
                todo.extend([(area_l, i, k, use_pf), (area_r, k, j, use_pf)])
                continue
        stop = True
        sec = ((inv2 >= i) & (inv2 < j))[inv1]
        ids = e0[sec].unique(return_inverse=True)[1]
        done.append((sec, ids))
        print('Section: %d >= deg >= %d, #nodes: %d, space: %d, use_pf: %s'
              % (maxd, _ds[0], n, area, use_pf))
    return done


class SwitchableAttention(nn.Module):
    def __init__(
        self, din, heads, dim_head, causal=False,
        nb_features=None, qkv_bias=False, dropout=0,
        in_norm=True, **kwargs,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.in_norm = in_norm
        dim = dim_head * heads
        self.fast_attention = FastAttention(
            dim_head, nb_features,
            causal=causal,
            no_projection=False,
            kernel_fn=nn.ReLU(),
            generalized_attention=False,
        )
        if in_norm:
            self.norm = nn.LayerNorm(din)
        self.to_q = nn.Linear(din, dim, bias=qkv_bias)
        self.to_k = nn.Linear(din, dim, bias=qkv_bias)
        self.to_v = nn.Linear(din, dim, bias=qkv_bias)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        pfm = self.fast_attention.nb_features
        self.pf_threshold = (pfm * (pfm + dim_head)) ** 0.5 + pfm

    def reset_parameters(self):
        if self.in_norm:
            self.norm.reset_parameters()
        self.to_q.reset_parameters()
        self.to_k.reset_parameters()
        self.to_v.reset_parameters()
        self.to_out.reset_parameters()

    def attention(self, x, mask=None, linear=False, **kwargs):
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            (self.to_q(x), self.to_k(x), self.to_v(x)))
        if linear:
            if mask is not None:
                v.masked_fill_(~mask[:, None, :, None], 0.)
            out = self.fast_attention(q, k, v)
            out = self.dropout(out)
        else:
            s = torch.einsum('bhmd,bhnd->bhmn', q, k)
            if mask is not None:
                s.masked_fill_(~mask[:, None, None, :], float('-inf'))
            a = torch.softmax(s * (k.shape[-1] ** -0.5), dim=-1)
            a = self.dropout(a)
            out = torch.einsum('bhmn,bhnd->bhmd', a, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if mask is not None:
            out = out[mask]
        out = self.to_out(out)
        return out

    def forward(self, z, src, groups=None):
        if self.in_norm:
            z = self.norm(z)
        if groups is None:
            x, mask = to_dense_batch(z, src)
            h = self.attention(x, mask, linear=x.shape[1] > self.pf_threshold)
        else:
            h = z.new_zeros(z.shape[0], self.heads * self.dim_head)
            for group, ids in groups:
                x, mask = to_dense_batch(z[group], ids)
                h[group] = self.attention(
                    x, mask, linear=x.shape[1] > self.pf_threshold)
        return h


class NT2(nn.Module):
    def __init__(
            self, node_din=0, edge_din=0, incidence_din=0, dout=0,
            hidden=8, heads=4, n_layers=2, dropout=0, input_dropout=0,
            divide_factor=0.4, pf_threshold=-1,
            **kwargs):
        super(self.__class__, self).__init__()
        self.edge_din = edge_din
        self.incidence_din = incidence_din
        self.node_din = node_din
        self.dout = dout
        self.divide_factor = divide_factor
        self.pf_threshold = pf_threshold
        self.dim = dim = hidden * heads
        assert node_din + edge_din + incidence_din
        assert dout
        if node_din:
            self.node_enc = nn.Sequential(
                nn.Dropout(input_dropout),
                nn.Linear(node_din, dim))
            self.node_conv = SwitchableAttention(
                dim, heads, hidden, dropout=dropout, **kwargs)
        if edge_din:
            self.edge_enc = nn.Sequential(
                nn.Dropout(input_dropout),
                nn.Linear(edge_din, dim))
            self.edge_conv = SwitchableAttention(
                dim, heads, hidden, dropout=dropout, **kwargs)
        if incidence_din:
            # NOTE: edge_attr is usually one_hot indices to get embeddings
            # If edge_attr is embeddings, dropout first.
            self.incidence_enc = nn.Sequential(
                nn.Linear(incidence_din, dim),
                nn.GELU(),
                nn.Dropout(input_dropout),
                nn.Linear(dim, dim))
        self.convs0 = nn.ModuleList([
            SwitchableAttention(dim, heads, hidden, dropout=dropout, **kwargs)
            for _ in range(n_layers)])
        self.convs1 = nn.ModuleList([
            SwitchableAttention(dim, heads, hidden, dropout=dropout, **kwargs)
            for _ in range(n_layers)])
        self.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dout))
        for conv0, conv1 in zip(self.convs0, self.convs1):
            if pf_threshold < 0:
                self.pf_threshold = conv0.pf_threshold
            else:
                conv0.pf_threshold = pf_threshold
                conv1.pf_threshold = pf_threshold

    def forward(self, data):
        # In data, e is hyperedges, x is nodes,
        # edge_index is E2V, edge_attr is incidence attributes

        # Neighbourhoods Partitioning
        if data.get('nt2_e0') is None:
            data.nt2_e0, data.nt2_order0 = data.edge_index[0].sort()
            # NOTE: may cause problems when divide_factor < 0 and
            # there exists isolated nodes.
            if self.divide_factor >= 0:
                data.nt2_groups0 = divide_edges(
                    data.nt2_e0, self.pf_threshold, self.divide_factor)
            data.nt2_e1, data.nt2_order1 = data.edge_index[1].sort()
            if self.divide_factor >= 0:
                data.nt2_groups1 = divide_edges(
                    data.nt2_e1, self.pf_threshold, self.divide_factor)
        o0, o1 = data.nt2_order0, data.nt2_order1

        # Incidence Encoding
        if self.incidence_din:
            h = self.incidence_enc(data.edge_attr)
        else:
            h = data.x.new_zeros(data.edge_index.shape[1], self.dim)
        if self.node_din:
            h[o0] = h[o0] + self.node_conv(
                self.node_enc(data.x)[data.edge_index[1, o0]],
                data.nt2_e0, data.get('nt2_groups0'))
        if self.edge_din:
            h[o1] = h[o1] + self.edge_conv(
                self.edge_enc(data.e)[data.edge_index[0, o1]],
                data.nt2_e1, data.get('nt2_groups1'))

        # Dual NT
        for convs0, convs1 in zip(self.convs0, self.convs1):
            h0 = convs0(h[o0], data.nt2_e0, data.get('nt2_groups0'))
            h1 = convs1(h[o1], data.nt2_e1, data.get('nt2_groups1'))
            h[o0] = h[o0] + h0
            h[o1] = h[o1] + h1

        return self.classifier(h)

    def reset_parameters(self):
        reset_parameters(self.convs0, self.convs1, self.classifier)
        if self.node_din:
            reset_parameters(self.node_enc, self.node_conv)
        if self.edge_din:
            reset_parameters(self.edge_enc, self.edge_conv)
        if self.incidence_din:
            reset_parameters(self.incidence_enc)
