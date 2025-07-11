# FC-HGNN Main Training and Evaluation Script
# Implements k-fold cross-validation for brain connectivity classification

import sys
import time
import torch
import numpy as np
import os
from opt import *
from metrics import accuracy, auc, prf, metrics
from dataload import dataloader
from model import fc_hgnn
from dataload import LabelSmoothingLoss
from dataload import Logger
from contextlib import nullcontext
from torch_geometric.data import Batch  # NEW

# Suppress torch dynamo warnings
os.environ['TORCHDYNAMO_DISABLE'] = '1'

if __name__ == '__main__':

    # Initialize configuration and logging
    opt = OptInit().initialize()

    # Set up dual logging (console + file)
    filename = opt.log_path
    log = Logger(filename)
    sys.stdout = log

    # Load brain connectivity data and demographics
    dl = dataloader()
    raw_features, y, nonimg, phonetic_score = dl.load_data()

    # Batch all subject graphs into a single PyG Batch for fast processing
    batched_graph = Batch.from_data_list(raw_features).to(opt.device, non_blocking=True)

    # Set up k-fold cross-validation splits
    n_folds = opt.n_folds
    cv_splits = dl.data_split(n_folds)

    # Initialize metric arrays for cross-validation results
    corrects = np.zeros(n_folds, dtype=np.int32) 
    accs = np.zeros(n_folds, dtype=np.float32) 
    sens = np.zeros(n_folds, dtype=np.float32) 
    spes = np.zeros(n_folds, dtype=np.float32) 
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds,3], dtype=np.float32)

    # Main cross-validation loop
    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold))

        # Debug limit for development (can skip later folds)
        if fold < 100:
            print("\r\n========================== Fold {} ==========================".format(fold))

            # Get train/test indices for current fold
            train_ind = cv_splits[fold][0]
            test_ind = cv_splits[fold][1]

            # Initialize model and labels for current fold
            # Fix tensor construction warning by using torch.from_numpy for numpy arrays
            if isinstance(y, np.ndarray):
                labels = torch.from_numpy(y).to(dtype=torch.long, device=opt.device)
            else:
                labels = torch.tensor(y, dtype=torch.long).to(opt.device)
            model = fc_hgnn(nonimg, phonetic_score, dl).to(opt.device)

            # Pre-compute population graph edges once (does not depend on embeddings)
            same_idx_np, diff_idx_np = dl.get_inputs(nonimg, torch.zeros((len(raw_features), 1)), phonetic_score)
            model.set_population_edges(same_idx_np, diff_idx_np)

            # ------------------------------------------------------------------
            # Optional graph compilation (PyTorch ≥ 2.0)
            # Using mode="reduce-overhead" tends to pay off for models dominated
            # by many small kernel launches while keeping compile time modest.
            # If anything goes wrong we silently fall back to eager execution.
            # ------------------------------------------------------------------
            if hasattr(torch, "compile"):
                try:
                    model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
                    print("[torch.compile] Enabled graph mode for extra speed-ups")
                except Exception as compile_err:
                    print(f"[torch.compile] Disabled – reason: {compile_err}")

            print(model)

            # Set up loss function (CrossEntropy or Label Smoothing)
            loss_fn = torch.nn.CrossEntropyLoss()
            # loss_fn =LabelSmoothingLoss()  # Alternative loss function

            # Initialize optimizer and model save path
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
            fold_model_path = os.path.join(opt.ckpt_path, "inffus_fold{}.pth".format(fold))

            def train():
                # Training function for current fold
                acc = 0
                # AMP utilities - Updated to use torch.amp.GradScaler
                scaler = torch.amp.GradScaler(enabled=bool(opt.amp))
                # Use the new torch.amp.autocast API (PyTorch ≥2.1). Fall back to a no-op context when AMP is disabled.
                autocast_ctx = (lambda: torch.amp.autocast(device_type='cuda')) if bool(opt.amp) else nullcontext
                for epoch in range(opt.num_iter):
                    # Training phase
                    model.train()
                    optimizer.zero_grad()
                    with autocast_ctx():
                        node_logits = model(batched_graph)
                        loss_cla = loss_fn(node_logits[train_ind], labels[train_ind])
                        loss = loss_cla

                    # Backward with scaler when AMP enabled
                    if bool(opt.amp):
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    correct_train, acc_train = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])

                    # Evaluation phase
                    model.eval()
                    with torch.set_grad_enabled(False):
                        with autocast_ctx():
                            node_logits = model(batched_graph)
                    logits_test = node_logits[test_ind].detach().cpu().numpy()
                    correct_test, acc_test = accuracy(logits_test, y[test_ind])
                    test_sen, test_spe = metrics(logits_test, y[test_ind])
                    auc_test = auc(logits_test,y[test_ind])
                    prf_test = prf(logits_test,y[test_ind])

                    # Print epoch results
                    print("Epoch: {},\tce loss: {:.5f},\tce loss_cla: {:.5f},\ttrain acc: {:.5f},\ttest acc: {:.5f},\ttest spe: {:.5f},\ttest sen: {:.5f}".format(epoch, loss.item(),loss_cla.item(),acc_train.item(),acc_test.item(),test_spe,test_sen),time.localtime(time.time()))

                    # Save best model based on test accuracy
                    if acc_test > acc:
                        acc = acc_test
                        correct = correct_test
                        aucs[fold] = auc_test
                        prfs[fold] = prf_test
                        sens[fold] = test_sen
                        spes[fold] = test_spe
                        if opt.ckpt_path != '':
                            if not os.path.exists(opt.ckpt_path):
                                os.makedirs(opt.ckpt_path)
                            torch.save(model.state_dict(), fold_model_path)
                            print("{} Saved model to:{}".format("\u2714", fold_model_path))

                # Store final fold results
                accs[fold] = acc
                corrects[fold] = correct
                print("\r\n => Fold {} test accuacry {:.5f}".format(fold, acc))


            def evaluate():
                # Evaluation function for pre-trained models
                print("  Number of testing samples %d" % len(test_ind))
                print('  Start testing...')
                model.load_state_dict(torch.load(fold_model_path))
                model.eval()
                node_logits = model(batched_graph)
                logits_test = node_logits[test_ind].detach().cpu().numpy()
                corrects[fold], accs[fold] = accuracy(logits_test, y[test_ind])
                sens[fold], spes[fold] = metrics(logits_test, y[test_ind])
                aucs[fold] = auc(logits_test, y[test_ind])
                prfs[fold] = prf(logits_test, y[test_ind])
                print("  Fold {} test accuracy {:.5f}, AUC {:.5f},".format(fold, accs[fold], aucs[fold]))

            # Run training or evaluation based on config
            if opt.train == 1:
                train()
            elif opt.train == 0:
                evaluate()

    # Print final cross-validation results
    print("\r\n========================== Finish ==========================") 
    n_samples = len(y)
    acc_nfold = np.sum(corrects)/n_samples
    print("=> Average test accuracy in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(accs),np.var(accs)))
    print("=> Average test sen in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(sens),np.var(sens)))
    print("=> Average test spe in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(spes),np.var(spes)))
    print("=> Average test AUC in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(aucs),np.var(aucs)))
    se, sp, f1 = np.mean(prfs, axis=0)
    se_var, sp_var, f1_var = np.var(prfs, axis=0)
    print("=> Average test sensitivity {:.4f}({:.4f}), specificity {:.4f}({:.4f}), F1-score {:.4f}({:.4f})".format(se,se_var, sp,sp_var, f1,f1_var))
    print("{} Saved model to:{}".format("\u2714", opt.ckpt_path))


