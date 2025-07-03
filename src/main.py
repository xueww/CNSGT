import argparse
import time
import torch
import numpy as np
from Models import get_model
import torch.nn.functional as F

from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import dill as pickle

import csv
import timeit
import os
import torch.nn as nn

def KLAnnealer(opt, epoch):
    beta = opt.KLA_ini_beta + opt.KLA_inc_beta * ((epoch + 1) - opt.KLA_beg_epoch)
    return beta

def loss_function(opt, beta, preds_prop, preds_mol, ys_cond, ys_mol, mu, log_var):
    RCE_mol = F.cross_entropy(preds_mol.contiguous().view(-1, preds_mol.size(-1)), ys_mol, ignore_index=opt.trg_pad, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    if opt.use_cond2dec == True:
        RCE_prop = F.mse_loss(preds_prop, ys_cond, reduction='sum')
        loss = RCE_mol + RCE_prop + beta * KLD
    else:
        RCE_prop = torch.zeros(1)
        loss = RCE_mol + beta * KLD
    return loss, RCE_mol, RCE_prop, KLD

def train_model(model, opt, SRC, TRG, robustScaler):
    print("training model...")
    model.train()

    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()

    beta = 0
    current_step = 0
    for epoch in range(opt.epochs):
        total_loss, RCE_mol_loss, RCE_prop_loss, KLD_loss= 0, 0, 0, 0
        total_loss_te, RCE_mol_loss_te, RCE_prop_loss_te, KLD_loss_te = 0, 0, 0, 0
        total_loss_accum_te, RCE_mol_loss_accum_te, RCE_prop_loss_accum_te, KLD_loss_accum_te = 0, 0, 0, 0
        accum_train_printevery_n, accum_test_n, accum_test_printevery_n = 0, 0, 0

        if opt.floyd is False:
            print("     {TR}   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')

        # KL annealing
        if opt.use_KLA == True:
            if epoch + 1 >= opt.KLA_beg_epoch and beta < opt.KLA_max_beta:
                beta = KLAnnealer(opt, epoch)
        else:
            beta = 1

        for i, batch in enumerate(opt.train):
            current_step += 1
            src = batch.src.transpose(0, 1).to('cuda')
            trg = batch.trg.transpose(0, 1).to('cuda')
            trg_input = trg[:, :-1]

            cond = torch.stack([batch.LogP, batch.HBD, batch.TPSA, batch.MW, batch.pKa, batch.LogD]).transpose(0, 1).to('cuda')

            
            src_mask, trg_mask = None, None  
            preds_prop, preds_mol, mu, log_var, z = model(src, trg_input, cond, src_mask, trg_mask)
            ys_mol = trg[:, 1:].contiguous().view(-1)
            ys_cond = torch.unsqueeze(cond, 2).contiguous().view(-1, opt.cond_dim, 1)

            opt.optimizer.zero_grad()

            loss, RCE_mol, RCE_prop, KLD = loss_function(opt, beta, preds_prop, preds_mol, ys_cond, ys_mol, mu, log_var)

            loss.backward()
            opt.optimizer.step()
            if opt.lr_scheduler == "SGDR":
                opt.sched.step()

            if opt.lr_scheduler == "WarmUpDefault":
                head = np.float(np.power(np.float(current_step), -0.5))
                tail = np.float(current_step) * np.power(np.float(opt.lr_WarmUpSteps), -1.5)
                lr = np.float(np.power(np.float(opt.d_model), -0.5)) * min(head, tail)
                for param_group in opt.optimizer.param_groups:
                    param_group['lr'] = lr

            for param_group in opt.optimizer.param_groups:
                current_lr = param_group['lr']

            total_loss += loss.item()
            RCE_mol_loss += RCE_mol.item()
            RCE_prop_loss += RCE_prop.item()
            KLD_loss += KLD.item()

            accum_train_printevery_n += len(batch)
            if (i + 1) % opt.printevery == 0:
                 p = int(100 * (i + 1) / opt.train_len)
                 avg_loss = total_loss /accum_train_printevery_n
                 avg_RCE_mol_loss = RCE_mol_loss /accum_train_printevery_n
                 avg_RCE_prop_loss = RCE_prop_loss /accum_train_printevery_n
                 avg_KLD_loss = KLD_loss /accum_train_printevery_n
                 if opt.floyd is False:
                    print("     {TR}   %dm: epoch %d [%s%s]  %d%%  loss = %.3f, RCE_mol = %.3f, RCE_prop = %.5f, KLD = %.5f, beta = %.4f, lr = %.6f" % ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss, avg_RCE_mol_loss, avg_RCE_prop_loss, avg_KLD_loss, beta, current_lr), end='\r')
                 else:
                    print("     {TR}   %dm: epoch %d [%s%s]  %d%%  loss = %.3f, RCE_mol = %.3f, RCE_prop = %.5f, KLD = %.5f, beta = %.4f, lr = %.6f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss, avg_RCE_mol_loss, avg_RCE_prop_loss, avg_KLD_loss, beta, current_lr))
                 accum_train_printevery_n, total_loss, RCE_mol_loss, RCE_prop_loss, KLD_loss = 0, 0, 0, 0, 0
            
        print("     {TR}   %dm: epoch %d [%s%s]  %d%%  loss = %.3f, RCE_mol = %.3f, RCE_prop = %.5f, KLD = %.5f, beta = %.4f, lr = %.6f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, avg_RCE_mol_loss, avg_RCE_prop_loss, avg_KLD_loss, beta, current_lr))

       
        if opt.imp_test == True:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(opt.test):
                    src = batch.src.transpose(0, 1).to('cuda')
                    trg = batch.trg.transpose(0, 1).to('cuda')
                    trg_input = trg[:, :-1]
                    cond = torch.stack([batch.LogP, batch.HBD, batch.TPSA, batch.MW, batch.pKa, batch.LogD]).transpose(0, 1).to('cuda')

                    # src_mask, trg_mask = create_masks(src, trg_input, cond, opt)
                    src_mask, trg_mask = None, None  # 简版：未实现 create_masks
                    preds_prop, preds_mol, mu, log_var, z = model(src, trg_input, cond, src_mask, trg_mask)
                    ys_mol = trg[:, 1:].contiguous().view(-1)
                    ys_cond = torch.unsqueeze(cond, 2).contiguous().view(-1, opt.cond_dim, 1)

                    loss_te, RCE_mol_te, RCE_prop_te, KLD_te = loss_function(opt, beta, preds_prop, preds_mol, ys_cond, ys_mol, mu, log_var)

                    total_loss_te += loss_te.item()
                    RCE_mol_loss_te += RCE_mol_te.item()
                    RCE_prop_loss_te += RCE_prop_te.item()
                    KLD_loss_te += KLD_te.item()
                    total_loss_accum_te += loss_te.item()
                    RCE_mol_loss_accum_te += RCE_mol_te.item()
                    RCE_prop_loss_accum_te += RCE_prop_te.item()
                    KLD_loss_accum_te += KLD_te.item()

                    accum_test_n += len(batch)
                    accum_test_printevery_n += len(batch)
                    if (i + 1) % opt.printevery == 0:
                        p = int(100 * (i + 1) / opt.test_len)
                        avg_loss_te = total_loss_te /accum_test_printevery_n
                        avg_RCE_mol_loss_te = RCE_mol_loss_te /accum_test_printevery_n
                        avg_RCE_prop_loss_te = RCE_prop_loss_te /accum_test_printevery_n
                        avg_KLD_loss_te = KLD_loss_te /accum_test_printevery_n
                        if opt.floyd is False:
                            print("     {TE}   %dm:         [%s%s]  %d%%  loss = %.3f, RCE_mol = %.3f, RCE_prop = %.5f, KLD = %.5f, beta = %.4f" % \
                            ((time.time() - start) // 60, "".join('#' * (p // 5)), "".join(' ' * (20 - (p // 5))), p, avg_loss_te, avg_RCE_mol_loss_te, avg_RCE_prop_loss_te, avg_KLD_loss_te, beta), end='\r')
                        else:
                            print("     {TE}   %dm:         [%s%s]  %d%%  loss = %.3f, RCE_mol = %.3f, RCE_prop = %.3f, KLD = %.5f, beta = %.4f" % \
                            ((time.time() - start) // 60, "".join('#' * (p // 5)), "".join(' ' * (20 - (p // 5))), p, avg_loss_te, avg_RCE_mol_loss_te, avg_RCE_prop_loss_te, avg_KLD_loss_te, beta))
                        accum_test_printevery_n, total_loss_te, RCE_mol_loss_te, RCE_prop_loss_te, KLD_loss_te = 0, 0, 0, 0, 0
                print("     {TE}   %dm:         [%s%s]  %d%%  loss = %.3f, RCE_mol = %.3f, RCE_prop = %.5f, KLD = %.5f, beta = %.4f\n" % \
                            ((time.time() - start) // 60, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100, avg_loss_te, avg_RCE_mol_loss_te, avg_RCE_prop_loss_te, avg_KLD_loss_te, beta))


class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(SimpleTransformerModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.trg_tok_emb = nn.Embedding(trg_vocab_size, d_model)
        self.generator = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg, *args, **kwargs):
        src_emb = self.src_tok_emb(src)
        trg_emb = self.trg_tok_emb(trg)
        output = self.transformer(src_emb, trg_emb)
        return self.generator(output)

def main():
    parser = argparse.ArgumentParser()
    # 只保留实际在代码中 opt.xxx 被引用的参数
    parser.add_argument('-imp_test', type=bool, default=False)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-no_cuda', type=str, default=False)
    parser.add_argument('-lr_scheduler', type=str, default="WarmUpDefault", help="WarmUpDefault, SGDR")
    parser.add_argument('-lr_WarmUpSteps', type=int, default=8000, help="only for WarmUpDefault")
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-lr_beta1', type=float, default=0.9)
    parser.add_argument('-lr_beta2', type=float, default=0.98)
    parser.add_argument('-lr_eps', type=float, default=1e-9)
    parser.add_argument('-use_KLA', type=bool, default=True)
    parser.add_argument('-KLA_ini_beta', type=float, default=0.02)
    parser.add_argument('-KLA_inc_beta', type=float, default=0.02)
    parser.add_argument('-KLA_max_beta', type=float, default=1.0)
    parser.add_argument('-KLA_beg_epoch', type=int, default=1)
    parser.add_argument('-use_cond2dec', type=bool, default=False)
    parser.add_argument('-latent_dim', type=int, default=128)
    parser.add_argument('-cond_dim', type=int, default=6)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.3)
    parser.add_argument('-batchsize', type=int, default=1024*6)
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-save_folder_name', type=str, default='cns_saved_model')
    parser.add_argument('-printevery', type=int, default=5)
    parser.add_argument('-historyevery', type=int, default=5)
    parser.add_argument('-load_weights')
    parser.add_argument('-checkpoint', type=int, default=0)
    # 其它未被引用参数已删除
    opt = parser.parse_args()
    # 假设 SRC/TRG vocab size 由数据集决定，这里用示例数值
    src_vocab_size = 1000
    trg_vocab_size = 1000
    model = TransformerModel(src_vocab_size, trg_vocab_size, d_model=opt.d_model, nhead=opt.heads, num_encoder_layers=opt.n_layers, num_decoder_layers=opt.n_layers, dropout=opt.dropout)
   
    train_model(model, opt, None, None, None)


if __name__ == "__main__":
    main()

