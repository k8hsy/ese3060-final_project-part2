import torch
import torch.optim

class ScheduleFreeAdamW(torch.optim.Optimizer):
    """
    Schedule-Free AdamW Optimizer.
    Based on "The Road Less Scheduled" (Defazio et al., 2024).
    """
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, warmup_steps=1000, r=0.0, weight_lr_power=2.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, r=r, k=0,
                        warmup_steps=warmup_steps, train_mode=True,
                        weight_sum=0.0, lr_max=-1.0, weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    def eval(self):
        for group in self.param_groups:
            group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
            decay = group['weight_decay']
            k = group['k']
            r = group['r']
            warmup_steps = group['warmup_steps']
            
            if k == 0:
                group['lr_max'] = group['lr']

            # Schedule-free logic: Effective LR depends on time k (1/sqrt(k) decay implicitly)
            if k < warmup_steps:
                sched = (k + 1) / warmup_steps
            else:
                sched = 1.0

            bias_correction2 = 1 - beta2 ** (k + 1)
            lr = group['lr_max'] * sched * (bias_correction2**0.5)

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['z'] = p.clone() # Primal (optimization variable)
                    state['exp_avg_sq'] = torch.zeros_like(p) # Second moment

                z = state['z']
                exp_avg_sq = state['exp_avg_sq']

                # Weight decay on the "z" variable (the energetic particle)
                if decay != 0:
                    grad = grad.add(z, alpha=decay)

                # Adam-style Second Moment Update
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                denom = exp_avg_sq.sqrt().add_(eps)

                # Update Z (The "energetic" variable)
                z.addcdiv_(grad, denom, value=-lr)

                # Update X (The "averaged" variable - what we use for Eval)
                # x_{k+1} = (1-c)x_k + c z_{k+1}
                # The weight c is derived from 1/(k+1)
                ck = 1 / (k + 1 + r)
                p.mul_(1 - ck).add_(z, alpha=ck)

            group['k'] = k + 1
        return loss