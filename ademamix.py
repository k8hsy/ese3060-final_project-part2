import torch
from torch.optim.optimizer import Optimizer

class AdEMAMix(Optimizer):
    """
    Implements AdEMAMix algorithm (Apple, 2024).
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999, 0.9999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 0)
        alpha (float, optional): coefficient for the slow-moving average (default: 5.0)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), eps=1e-8,
                 weight_decay=0, alpha=5.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, alpha=alpha)
        super(AdEMAMix, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_slows = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('AdEMAMix does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Slow exponential moving average
                        state['exp_avg_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    exp_avg_slows.append(state['exp_avg_slow'])
                    
                    state['step'] += 1
                    state_steps.append(state['step'])

            beta1, beta2, beta3 = group['betas']
            alpha = group['alpha']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                exp_avg_slow = exp_avg_slows[i]
                step = state_steps[i]

                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)

                denom = (exp_avg_sq.sqrt() / (1 - beta2 ** step)**0.5).add_(eps)
                
                # AdEMAMix update rule: combination of fast (exp_avg) and slow (exp_avg_slow)
                update = (exp_avg + alpha * exp_avg_slow) / (1 - beta1 ** step)
                
                param.addcdiv_(update, denom, value=-lr)

        return loss