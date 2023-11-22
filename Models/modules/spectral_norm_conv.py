"""
Spectral Normalization from https://arxiv.org/abs/1802.05957
with the modification for Conv layer: https://openreview.net/forum?id=JCRblSgs34Z
"""

import math
import torch
from torch.nn.functional import normalize
from typing import Any, Optional, TypeVar
from torch.nn import Module
from torch.linalg import matrix_norm
from Utils.console import console

__all__ = ['SpectralNormConv', 'SpectralNormConvLoadStateDictPreHook', 'SpectralNormConvStateDictHook',
           'spectral_norm_conv', 'remove_spectral_norm_conv']
T_module = TypeVar('T_module', bound=Module)

class SpectralNormConv:
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version: int = 1
    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.
    name: str
    dim: int
    n_power_iterations: int
    eps: float
    thres: float

    def __init__(self, name: str = 'weight', n_power_iterations: int = 50, dim: int = 0, eps: float = 1e-12, debug:bool=False, thres:float=1.0) -> None:
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             f'got n_power_iterations={n_power_iterations}')
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.debug = debug
        self.thres = thres

    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)
    
    def l2_normalize(self, tensor, eps=1e-12):
        norm = float(torch.sqrt(torch.sum(tensor.float() * tensor.float())))
        norm = max(norm, eps)
        ans = tensor / norm
        return ans

    def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor:

        weight = getattr(module, self.name + '_orig')

        _, _, h, w = weight.shape

        u1 = getattr(module, self.name + '_u1')
        u2 = getattr(module, self.name + '_u2')
        u3 = getattr(module, self.name + '_u3')
        u4 = getattr(module, self.name + '_u4')

        v1 = getattr(module, self.name + '_v1')
        v2 = getattr(module, self.name + '_v2')
        v3 = getattr(module, self.name + '_v3')
        v4 = getattr(module, self.name + '_v4')
        
        mat1, mat2, mat3, mat4 = _conv_matrices(conv_filter=weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v1 = normalize(torch.mv(mat1.t(), u1), dim=0, eps=self.eps, out=v1)
                    u1 = normalize(torch.mv(mat1, v1), dim=0, eps=self.eps, out=u1)

                    v2 = normalize(torch.mv(mat2.t(), u2), dim=0, eps=self.eps, out=v2)
                    u2 = normalize(torch.mv(mat2, v2), dim=0, eps=self.eps, out=u2)

                    v3 = normalize(torch.mv(mat3.t(), u3), dim=0, eps=self.eps, out=v3)
                    u3 = normalize(torch.mv(mat3, v3), dim=0, eps=self.eps, out=u3)

                    v4 = normalize(torch.mv(mat4.t(), u4), dim=0, eps=self.eps, out=v4)
                    u4 = normalize(torch.mv(mat4, v4), dim=0, eps=self.eps, out=u4)


                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u1 = u1.clone(memory_format=torch.contiguous_format)
                    v1 = v1.clone(memory_format=torch.contiguous_format)
                    
                    u2 = u2.clone(memory_format=torch.contiguous_format)
                    v2 = v2.clone(memory_format=torch.contiguous_format)

                    u3 = u3.clone(memory_format=torch.contiguous_format)
                    v3 = v3.clone(memory_format=torch.contiguous_format)

                    u4 = u4.clone(memory_format=torch.contiguous_format)
                    v4 = v4.clone(memory_format=torch.contiguous_format)

        sigma1 = torch.dot(u1, torch.mv(mat1, v1))
        sigma2 = torch.dot(u2, torch.mv(mat2, v2))
        sigma3 = torch.dot(u3, torch.mv(mat3, v3))
        sigma4 = torch.dot(u4, torch.mv(mat4, v4))

        sigma = math.sqrt(h*w) * torch.min(sigma1, torch.min(sigma2, torch.min(sigma3, sigma4))).item()
        c = min(1, self.thres / (sigma + 1e-12))
        weight = weight * c
        return weight

    def remove(self, module: Module) -> None:
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)

        delattr(module, self.name + '_u1')
        delattr(module, self.name + '_u2')
        delattr(module, self.name + '_u3')
        delattr(module, self.name + '_u4')

        delattr(module, self.name + '_v1')
        delattr(module, self.name + '_v2')
        delattr(module, self.name + '_v3')
        delattr(module, self.name + '_v4')
        delattr(module, self.name + '_orig')

        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        v = torch.linalg.multi_dot([weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)]).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float, debug: bool, thres: float) -> 'SpectralNormConv':
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, SpectralNormConv) and hook.name == name:
                raise RuntimeError(f"Cannot register two spectral_norm hooks on the same parameter {name}")

        fn = SpectralNormConv(name, n_power_iterations, dim, eps, debug=debug, thres=thres)
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(f'`SpectralNorm` cannot be applied as parameter `{name}` is None')
        if isinstance(weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError(
                'The module passed to `SpectralNorm` can\'t have uninitialized parameters. '
                'Make sure to run the dummy forward before applying spectral normalization')

        with torch.no_grad():
            # weight_mat = fn.reshape_weight_to_matrix(weight)

            _, _, h, w = weight.shape
            mat1, mat2, mat3, mat4 = _conv_matrices(conv_filter=weight)
            # randomly initialize `u` and `v`

            u1 = normalize(weight.new_empty(mat1.shape[0]).normal_(0, 1), dim=0, eps=fn.eps)
            v1 = normalize(weight.new_empty(mat1.shape[1]).normal_(0, 1), dim=0, eps=fn.eps)

            u2 = normalize(weight.new_empty(mat2.shape[0]).normal_(0, 1), dim=0, eps=fn.eps)
            v2 = normalize(weight.new_empty(mat2.shape[1]).normal_(0, 1), dim=0, eps=fn.eps)

            u3 = normalize(weight.new_empty(mat3.shape[0]).normal_(0, 1), dim=0, eps=fn.eps)
            v3 = normalize(weight.new_empty(mat3.shape[1]).normal_(0, 1), dim=0, eps=fn.eps)

            u4 = normalize(weight.new_empty(mat4.shape[0]).normal_(0, 1), dim=0, eps=fn.eps)
            v4 = normalize(weight.new_empty(mat4.shape[1]).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u1", u1)
        module.register_buffer(fn.name + "_v1", v1)
        
        module.register_buffer(fn.name + "_u2", u2)
        module.register_buffer(fn.name + "_v2", v2)

        module.register_buffer(fn.name + "_u3", u3)
        module.register_buffer(fn.name + "_v3", v3)
        
        module.register_buffer(fn.name + "_u4", u4)
        module.register_buffer(fn.name + "_v4", v4)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormConvStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormConvLoadStateDictPreHook(fn))
        return fn

class SpectralNormConvLoadStateDictPreHook:
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn) -> None:
        self.fn = fn

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs) -> None:
        fn = self.fn
        version = local_metadata.get('spectral_norm_conv', {}).get(fn.name + '.version', None)
        if version is None or version < 1:
            weight_key = prefix + fn.name
            if version is None and all(weight_key + s in state_dict for s in ('_orig', '_u1', '_v1', '_u2', '_v2', '_u3', '_v3', '_u4', '_v4')) and \
                    weight_key not in state_dict:
                # Detect if it is the updated state dict and just missing metadata.
                # This could happen if the users are crafting a state dict themselves,
                # so we just pretend that this is the newest.
                return
            has_missing_keys = False
            for suffix in ('_orig', '', '_u1', '_u2', '_u3', '_u4'):
                key = weight_key + suffix
                if key not in state_dict:
                    has_missing_keys = True
                    if strict:
                        missing_keys.append(key)
            if has_missing_keys:
                return
            # with torch.no_grad():
            #     weight_orig = state_dict[weight_key + '_orig']
            #     weight = state_dict.pop(weight_key)
            #     sigma = (weight_orig / weight).mean()
            #     weight_mat = fn.reshape_weight_to_matrix(weight_orig)
            #     u = state_dict[weight_key + '_u']
            #     v = fn._solve_v_and_rescale(weight_mat, u, sigma)
            #     state_dict[weight_key + '_v'] = v

class SpectralNormConvStateDictHook:
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata) -> None:
        if 'spectral_norm_conv' not in local_metadata:
            local_metadata['spectral_norm_conv'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['spectral_norm_conv']:
            raise RuntimeError(f"Unexpected key in metadata['spectral_norm_conv']: {key}")
        local_metadata['spectral_norm_conv'][key] = self.fn._version

def _conv_matrices(conv_filter: torch.Tensor):

    out_ch, in_ch, h, w = conv_filter.shape
    
    transpose1 = torch.transpose(conv_filter, 1, 2)
    matrix1 = transpose1.reshape(out_ch*h, in_ch*w)
    
    transpose2 = torch.transpose(conv_filter, 1, 3)
    matrix2 = transpose2.reshape(out_ch*w, in_ch*h)

    matrix3 = conv_filter.view(out_ch, in_ch*h*w)

    transpose4 = torch.transpose(conv_filter, 0, 1)
    matrix4 = transpose4.reshape(in_ch, out_ch*h*w)

    return matrix1, matrix2, matrix3, matrix4

def spectral_norm_conv(module: T_module,
                  name: str = 'weight',
                  n_power_iterations: int = 100,
                  eps: float = 1e-12,
                  dim: Optional[int] = None, 
                  debug:bool=False, 
                  thres:float=1.0) -> T_module:
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    .. note::
        This function has been reimplemented as
        :func:`torch.nn.utils.parametrizations.spectral_norm` using the new
        parametrization functionality in
        :func:`torch.nn.utils.parametrize.register_parametrization`. Please use
        the newer version. This function will be deprecated in a future version
        of PyTorch.

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNormConv.apply(module, name, n_power_iterations, dim, eps, debug=debug, thres=thres)
    return module

def remove_spectral_norm_conv(module: T_module, name: str = 'weight') -> T_module:
    r"""Removes the spectral normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNormConv) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    else:
        raise ValueError(f"spectral_norm of '{name}' not found in {module}")

    for k, hook in module._state_dict_hooks.items():
        if isinstance(hook, SpectralNormConvStateDictHook) and hook.fn.name == name:
            del module._state_dict_hooks[k]
            break

    for k, hook in module._load_state_dict_pre_hooks.items():
        if isinstance(hook, SpectralNormConvLoadStateDictPreHook) and hook.fn.name == name:
            del module._load_state_dict_pre_hooks[k]
            break

    return module
