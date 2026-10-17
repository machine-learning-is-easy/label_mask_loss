import torch
import math
# Define custom optimizer
class LWADAM(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0):
        super().__init__(params, lr=lr, betas=betas)
        self.weight_decay = weight_decay
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.zero = torch.tensor(0.0).to(self.device)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if self.weight_decay != 0:
                    grad = grad.add(p.data, alpha=self.weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                # filter the step_size
                max_exp_avg = torch.max(torch.abs(exp_avg))

                mask = torch.abs(exp_avg) > max_exp_avg/10
                exp_avg = torch.where(mask, exp_avg, self.zero)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
