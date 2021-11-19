from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch


def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.predict_TD = hparams['predict_TD']
        self.optimizer_spec = optimizer_spec

        # <DONE>: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        # WARNING: Make sure you use different types of weight 
        #          initializations for these two functions

        # HINT 1) Check out the method ptu.build_mlp
        # HINT 2) There are two weight init methods defined above
        if not self.predict_TD:
            self.f = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, self.size, init_method=init_method_1)
            self.f_hat = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, self.size, init_method=init_method_2)
        else:
            self.f_hat = ptu.build_mlp(self.ob_dim + self.ac_dim, self.ob_dim, self.n_layers, self.size, init_method=init_method_1)
        
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        if not self.predict_TD:
            self.f.to(ptu.device)
        self.f_hat.to(ptu.device)

    def forward(self, ob_no, ac_na=None, next_ob_no=None):
        # <DONE>: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        if not self.predict_TD:
            targets = self.f(ob_no).detach()
            predictions = self.f_hat(ob_no)
            return torch.norm(predictions - targets, dim=1)
        else:
            actions = torch.zeros((ob_no.shape[0], self.ac_dim))
            ac_na = ac_na.long()
            for i in range(ob_no.shape[0]):
                actions[i, ac_na[i]] = 1.0
            actions = actions.to(ob_no.device)
            input_ = torch.cat([ob_no, actions], dim=1)
            predictions = self.f_hat(input_)
            assert predictions.shape == next_ob_no.shape
            return torch.norm(predictions - next_ob_no, dim=1)

    def forward_np(self, ob_no, ac_na=None, next_ob_no=None):
        if not self.predict_TD:
            ob_no = ptu.from_numpy(ob_no)
            error = self(ob_no)
        else:
            ob_no = ptu.from_numpy(ob_no)
            ac_na = ptu.from_numpy(ac_na)
            next_ob_no = ptu.from_numpy(next_ob_no)
            error = self(ob_no, ac_na, next_ob_no)
        return ptu.to_numpy(error)

    # def forward_TD(self, ob_no, ac_na, next_ob_no):
    #     predictions = self.f_hat(torch.cat([ob_no, ac_na], dim=1))
    #     assert predictions.shape == next_ob_no.shape
    #     return torch.norm(predictions - next_ob_no, dim=1)

    # def forward_TD_np(self, ob_no, ac_na, next_ob_no):
    #     ob_no = ptu.from_numpy(ob_no)
    #     ac_na = ptu.from_numpy(ac_na)
    #     next_ob_no = ptu.from_numpy(next_ob_no)
    #     error = self.forward_TD(ob_no, ac_na, next_ob_no)
    #     return ptu.to_numpy(error)

    def update(self, ob_no, ac_na=None, next_ob_no=None):
        # <DONE>: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        if not self.predict_TD:
            prediction_errors = self(ptu.from_numpy(ob_no))
        else:
            ob_no = ptu.from_numpy(ob_no)
            ac_na = ptu.from_numpy(ac_na)
            next_ob_no = ptu.from_numpy(next_ob_no)
            prediction_errors = self(ob_no, ac_na, next_ob_no)
        
        loss = torch.mean(prediction_errors)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # def update_TD(self, ob_no, ac_na, next_ob_no):
    #     ob_no = ptu.from_numpy(ob_no)
    #     ac_na = ptu.from_numpy(ac_na)
    #     next_ob_no = ptu.from_numpy(next_ob_no)
    #     prediction_errors = self.forward_TD(ob_no, ac_na, next_ob_no)
    #     loss = torch.mean(prediction_errors)

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     return loss.item()
