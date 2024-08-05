import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

class StoLayer(nn.Module):
    def sto_init(self, 
                 n_components=4,
                 prior_mean=1.0, 
                 prior_std=0.40, 
                 post_mean_init=(1.0, 0.05), 
                 post_std_init=(0.40, 0.02),
                 mode='in'):
        self.prior_mean=nn.Parameter(torch.tensor(prior_mean), requires_grad=False)
        self.prior_std=nn.Parameter(torch.tensor(prior_std), requires_grad=False)

        # Flag to indicate stochaticity
        self.stochastic = 1

        # mode to inject stochaticity to the layer
        self.mode = mode

        # record parameter for repr
        self.n_components = n_components
        self.post_mean_init = post_mean_init
        self.post_std_init = post_std_init

        # latent_shape=[feature_size, 1, 1]
        # self.weight=[out_planes, in_planes, kernel_height, kernel_width] for Conv2d
        latent_shape = [1] * (self.weight.ndim-1)
        if mode =='in':
            latent_shape[0] = self.weight.shape[1]
        else:
            raise(NotImplementedError)    

        # post_mean and post_std are params shared over batch
        self.post_mean = nn.Parameter(torch.ones(n_components, *latent_shape), requires_grad=True)
        self.post_std = nn.Parameter(torch.ones(n_components, *latent_shape), requires_grad=True)

        # use a hierarchical Gaussian, that is, self.post_mean & self.post_std follows gaussian
        nn.init.normal_(self.post_mean, post_mean_init[0], post_mean_init[1])
        nn.init.normal_(self.post_std, post_std_init[0], post_std_init[1])

    def get_mul_noise(self, input, indices, stochastic_mode=1):
        """
        sample noise multiplied to feature
        stochastic_mode: 1 for random noise and 2 for fixed noise
        indices: indice for Gaussian component, for example, batch_size=8, component=4, then indices=[1,2,3,4,1,2,3,4]
        """
        mean = self.post_mean
        std = F.softplus(self.post_std)
        if stochastic_mode == 1:
            epsilon = torch.randn_like(input, device=input.device, dtype=input.dtype)
            noise = mean[indices] + std[indices] * epsilon
        elif stochastic_mode == 2:
            noise = mean[indices] 
        else:
            raise ValueError(f'stochastic mode {stochastic_mode} not supported')
        # mean.shape[0] will be automatically extended to the same size of indices
        return noise
    
    def _entropy_lower_bound(self, mean, std):
        """
        calculate entropy lower bound for mixed Gaussian
        mean, std: [n_components, *shape_of_feature]
        formula reference: https://www.mdpi.com/1099-4300/19/7/361
        """
        cond_entropy = D.Normal(mean, std).entropy().mean(dim=0).sum()

        # pairwise_mean_dist = [component_index1, component_index2, size_of_feature]
        mean_flat = mean.flatten(1)
        std_flat = std.flatten(1)
        var_flat = std_flat.square()
        logstd = std_flat.log().sum(1)

        pairwise_mean_dist = mean_flat.unsqueeze(1) - mean_flat.unsqueeze(0)
        pairwise_var_sum = var_flat.unsqueeze(1) + var_flat.unsqueeze(0)
        pairwise_std_logprod = logstd.unsqueeze(1) + logstd.unsqueeze(0)

        # calculation of chernoff alpha divergence
        c_a_diverge = (pairwise_mean_dist.square()/pairwise_var_sum).sum(2)/4 +\
                        0.5*(torch.log(0.5*pairwise_var_sum).sum(2)-pairwise_std_logprod)
        second_part = torch.logsumexp(-c_a_diverge, dim=1) - torch.log(torch.tensor(mean.size(0), dtype=mean.dtype, device=mean.device))

        entropy_lower_bound = cond_entropy - second_part.mean(0)
        return entropy_lower_bound


    def _kl(self, type='upper_bound'):
        """
        estimate kl divergence between mixed Gaussian and Gaussian by different strategy
        """
        prior = D.Normal(self.prior_mean, self.prior_std)
        if type == 'mean':
            post_mean = self.post_mean.mean(dim=0)
            post_std = F.softplus(self.post_std).square().sum().sqrt().mean(dim=0)
            post = D.Normal(post_mean, post_std)
            kl = D.kl_divergence(post, prior).sum()
        elif type == 'upper_bound':
            post_mean = self.post_mean
            post_std = F.softplus(self.post_std)
            post = D.Normal(post_mean, post_std)
            cross_entropy = (D.kl_divergence(post, prior) + post.entropy()).flatten(1).sum(1).mean()
            kl = cross_entropy - self._entropy_lower_bound(post_mean, post_std)
        else:
            raise NotImplementedError
        return kl

    def _entropy(self, type='upper_bound'):
        """
        estimate posterior entropy by different strategy
        """
        if type == 'upper_bound':
            mean = self.post_mean
            std = F.softplus(self.post_std)
            entropy = self._entropy_lower_bound(mean, std)
        else:
            raise NotImplementedError
        return entropy
    
    def to_determinstic(self):
        self.post_mean.requires_grad = False
        self.post_std.requires_grad = False
        self.stochastic = 0
    
    def to_stochastic(self, stochastic_mode=1):
        self.post_mean.requires_grad = True
        self.post_std.requires_grad = True
        self.stochastic = stochastic_mode

    def sto_extra_repr(self):
        return f"n_components={self.n_components}, prior_mean={self.prior_mean.detach().item()}, prior_std={self.prior_std.detach().item()}, posterior_mean_init={self.post_mean_init}, posterior_std_init={self.post_std_init}, mode={self.mode}, stochastic={self.stochastic}"

class StoLinear(nn.Linear, StoLayer):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool=False,
                 n_components=4, 
                 prior_mean=1.0, 
                 prior_std=0.40, 
                 post_mean_init=(1.0, 0.05), 
                 post_std_init=(0.40, 0.02), 
                 mode = 'in',
                 ):
        super().__init__(in_features, out_features, bias)
        self.sto_init(n_components, prior_mean, prior_std, post_mean_init, post_std_init, mode)

    def forward(self, x, indices):
        if self.stochastic == 0:
            pass
        if self.stochastic:
            noise = self.get_mul_noise(x, indices, stochastic_mode=self.stochastic)
            # noise.shape = [batch_size, in_features]
            if 'in' in self.mode:
                x = x * noise
            else:
                raise ValueError(f'{self.mode} not supported')
        x = super().forward(x)
        return x
    
    def extra_repr(self):
        return f'{super().extra_repr()}, {self.sto_extra_repr()}'
    
class StoConv2d(nn.Conv2d, StoLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 n_components=4,
                 prior_mean=1.0,
                 prior_std=0.40,
                 post_mean_init=(1.0, 0.05),
                 post_std_init=(0.40, 0.02),
                 mode='in',
                 ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.sto_init(n_components, prior_mean, prior_std, post_mean_init, post_std_init, mode)

    def forward(self, x, indices):
        """
        indices: indice for chosen Gaussian components
        """
        if self.stochastic == 0:
            pass
        if self.stochastic:
            noise = self.get_mul_noise(x, indices, stochastic_mode=self.stochastic)
            # noise.shape = [batch_size, in_features]
            if 'in' in self.mode:
                x = x*noise
            else:
                raise ValueError(f'{self.mode} not supported')

        x = super().forward(x)
        return x
    
    def extra_repr(self):
        return f'{super().extra_repr()}, {self.sto_extra_repr()}'

class StoModel(nn.Module):
    """
    Wrap stochastic layers and calculate loss
    """
    def sto_layers_init(self, n_components):
        self.n_components = n_components
        self.sto_layers = [m for m in self.modules() if isinstance(m, (StoLinear, StoConv2d))]
    
    def kl_and_entropy(self, kl_type='upper_bound'):
        kl = sum([m._kl(type=kl_type) for m in self.sto_layers])
        entropy = sum([m._entropy(type=kl_type) for m in self.sto_layers])
        return (kl, entropy)

    def vi_loss(self, y_pred, y_true, n_sample=1, kl_type='upper_bound', entropy_weight=.5):
        """
        y_true.shape=[batch_size]. y_pred.shape=[batch_size, n_sample, n_classes]
        """
        y_true = y_true.unsqueeze(1).expand(-1, n_sample)
        nll = -D.Categorical(logits=y_pred).log_prob(y_true).mean()
        kl, entropy = self.kl_and_entropy(kl_type=kl_type)
        kl_aug = kl - entropy_weight * entropy
        return (nll, kl_aug)     
    
    def to_determinstic(self):
        for layer in self.sto_layers:
            layer.to_determinstic()
    
    def to_stochastic(self, stochastic_mode):
        for layer in self.sto_layers:
            layer.to_stochastic(stochastic_mode=stochastic_mode)

