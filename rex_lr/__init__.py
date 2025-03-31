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
    def __init__(self, optimizer, max_val, min_val, num_epochs = 1, last_epoch=-1):
        self.num_epochs = num_epochs
        self.min_val = min_val
        self.max_val = max_val
        if  not self.min_val <= self.max_val:
            raise ValueError('Value of "min_val" should be less '
                             ' than value of "max_val". Got min_val='+str(min_val)+' and max_val='+str(max_val))
        

        super(REX_LR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        mod_iter = float(self.last_epoch % self.num_epochs)
        z = float(self.num_epochs- mod_iter) / self.num_epochs
        val = self.min_val + float(self.max_val - self.min_val) * (z / (1 - 0.9 + 0.9*z))
        return [val]
