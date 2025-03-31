# Copyright 2023 IvanVassi

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch


class REX_LR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_val, min_val, num_steps, last_epoch=-1):
        """
        Reflected Exponential (REX) schedule

        This scheduler implements a learning rate decay that follows a reciprocal
        exponential function, providing faster convergence compared to traditional
        decay methods.

        The learning rate varies between max_val and min_val according to a function
        that prioritizes higher learning rates early in training and gradually
        decreases as training progresses.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            max_val (float): Maximum learning rate value.
            min_val (float): Minimum learning rate value.
            num_steps (int): Total number of steps for the cycle.
            last_epoch (int, optional): The index of the last epoch. Default: -1.
        """
        self.num_steps = num_steps
        self.min_val = min_val
        self.max_val = max_val
        assert self.min_val <= self.max_val, (
            f'Value of "min_val" should be less than value of "max_val". Got min_val={min_val} and max_val={max_val}'
        )

        super(REX_LR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.num_steps:
            return [self.min_val]
        mod_iter = float(self.last_epoch % self.num_steps)
        z = float(self.num_steps - mod_iter) / self.num_steps
        val = self.min_val + float(self.max_val - self.min_val) * (z / (1 - 0.9 + 0.9 * z))
        return [val]


if __name__ == "__main__":
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters())
    lr_scheduler = REX_LR(optimizer, max_val=0.0005, min_val=0.00001, num_steps=100)
    for i in range(100):
        optimizer.step()
        print(lr_scheduler.get_lr())
        lr_scheduler.step()
    print(lr_scheduler.get_lr())
