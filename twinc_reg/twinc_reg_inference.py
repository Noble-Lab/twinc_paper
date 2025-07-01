"""
twinc_reg_inference.py
Author: Anupama Jha <anupamaj@uw.edu>
Inference for regression version of the
TwinC model.
"""

import torch
import pyfaidx
import argparse
import numpy as np
import configparser
from scipy.stats import pearsonr, spearmanr
from twinc_reg_network import TwinCRegNet
from trans_utils import count_pos_neg, decode_chrome_order_dict, decode_list, gc_predictor
from twinc_reg_train import extract_set_data, TwinCDataGenerator
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve


def run_twinc_reg_inference(config_file,
                            pred_set):
    """
    Run TwinC regression inference.
    :param config_file: str, path to the config file.
    :param pred_set: str, val or test
    :return: None
    """
    # read parameters from config file
    config = configparser.ConfigParser()
    config.read(config_file)

    chroms_order = decode_chrome_order_dict(
        config['data_parameters']['chroms_order'])
    print(f"chroms_order: {chroms_order}", flush=True)

    mean_expectation = float(config['data_parameters']['mean_expectation'])
    eps = float(config['data_parameters']['eps'])

    selected_resolution = int(config['data_parameters']['selected_resolution'])

    # hg38 sequence file
    sequences_file = config['input_files']['seq_file']

    # read sequence fasta file
    sequences = pyfaidx.Fasta(sequences_file)
    all_chroms_keys = sorted(sequences.keys())
    print(f"all_chroms_keys: {all_chroms_keys}", flush=True)

    # get chromsome lengths
    seq_chrom_lengths = []
    for chr_val in all_chroms_keys:
        len_value = len(sequences[chr_val][:].seq)
        print(f"{chr_val}, {len_value}", flush=True)
        seq_chrom_lengths.append(len_value)

    # get chromsome names
    seq_chrom_names = []
    for chr_val in all_chroms_keys:
        if 'chr' in chr_val:
            seq_chrom_names.append(chr_val)
        else:
            seq_chrom_names.append(f"chr{chr_val}")

    # get chromsome start and end index
    # in same order as the memory map
    seq_chrom_start = {}
    seq_chrom_end = {}
    cum_total = 0
    for i in range(len(all_chroms_keys)):
        seq_chrom_start[seq_chrom_names[i]] = cum_total
        cum_total += seq_chrom_lengths[i]
        seq_chrom_end[seq_chrom_names[i]] = cum_total - 1

    memmap_shape = (4, int(config["data_parameters"]["memmap_length"]))
    device = config["train_parameters"]["device"]

    hg38_memmap_path = config["input_files"]["seq_memmap"]

    # Load the genome into memory
    seq_memory_map = np.memmap(hg38_memmap_path,
                               dtype="float32",
                               mode="r",
                               shape=memmap_shape)

    save_prefix = config['output_files']['save_prefix']

    if pred_set == "validation":
        # Get the list of chromosomes for validation
        val_chroms = decode_list(config['data_parameters']['val_chroms'])
        # labels file
        labels_file = open(config['input_files']['labels_file_val'], 'r').readlines()

    elif pred_set == "test":
        # Get the list of chromosomes for test
        val_chroms = decode_list(config['data_parameters']['test_chroms'])
        # labels file
        labels_file = open(config['input_files']['labels_file_test'], 'r').readlines()

    else:
        print(f"Pred set can only be validation or test, enter correct option, exiting...")
        exit(1)

    # print(f"labels_file: {labels_file.shape}", flush=True)
    val_loci, reg_targets = extract_set_data(labels_file, val_chroms, seq_chrom_start, mean_expectation, eps)
    print(f"val_chroms: {val_chroms}, val_loci: {len(val_loci)}", flush=True)

    count_pos_neg(labels=np.array(val_loci[:, 6], dtype=int), set_name=f"{pred_set} set")

    # path to save best model
    best_save_model = config["model_parameters"]["best_model_path"]

    final_save_model = config["model_parameters"]["final_model_path"]

    # number of threads to process training data fetch
    num_workers = int(config["train_parameters"]["num_workers"])

    # batch size of training data
    batch_size = int(config["train_parameters"]["batch_size"])

    rep_name = config["data_parameters"]["rep_name"]

    val_gen = TwinCDataGenerator(seq_memory_map,
                                 val_loci,
                                 reg_targets,
                                 seq_chrom_start,
                                 seq_chrom_end,
                                 in_window=selected_resolution,
                                 reverse_complement=False,
                                 random_state=None
                                 )

    # Wrap it in a data loader
    val_gen = torch.utils.data.DataLoader(
        val_gen,
        pin_memory=True,
        num_workers=32,  # num_workers,
        batch_size=32,  # batch_size,
        shuffle=False,
    )
    print(f"best_save_model: {best_save_model}")
    # Move the model to appropriate device
    model = TwinCRegNet()
    # model.load_state_dict(torch.load(best_save_model))
    model.load_state_dict(torch.load(final_save_model))
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        # y_valid = torch.empty((0, 1))
        # valid_preds = torch.empty((0, 1)).to(device)
        y_valid = torch.empty((0, 5, 5))
        valid_preds = torch.empty((0, 5, 5)).to(device)
        gc_preds = torch.empty((0, 1)).to(device)
        gc_preds_diff = torch.empty((0, 1)).to(device)
        for cnt, data in enumerate(val_gen):
            if cnt % 1000 == 0:
                print(cnt, flush=True)
            # Get features and label batch
            X1, X2, y = data
            y_valid = torch.cat((y_valid, y))
            # Convert them to float
            X1, X2, y = X1.float(), X2.float(), y.float()
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)

            gc_pred, gc_pred_diff = gc_predictor(X1, X2, is_torch=True)
            gc_preds = torch.cat((gc_preds, gc_pred))
            gc_preds = gc_preds.to(device)

            gc_preds_diff = torch.cat((gc_preds_diff, gc_pred_diff))
            gc_preds_diff = gc_preds_diff.to(device)
            # Run forward pass
            val_pred = model.forward(X1, X2)
            valid_preds = torch.cat((valid_preds, val_pred))
            valid_preds = valid_preds.to(device)

        valid_preds, y_valid = valid_preds.to(device), y_valid.to(device)

        # compute cross_entropy loss for the validation set.
        mse_loss = ((valid_preds[~torch.isnan(y_valid)] - y_valid[~torch.isnan(y_valid)]) ** 2).mean()

        # Extract the validation loss
        valid_loss = mse_loss.detach().cpu().numpy()

        # Compute pearsonr
        pearsonr_twinc = pearsonr(valid_preds[~torch.isnan(y_valid)].detach().cpu().numpy(),
                                  y_valid[~torch.isnan(y_valid)].detach().cpu().numpy())[0]

        # Compute spearmanr
        spearmanr_twinc = spearmanr(valid_preds[~torch.isnan(y_valid)].detach().cpu().numpy(),
                                    y_valid[~torch.isnan(y_valid)].detach().cpu().numpy())[0]

        np.savez(f"{save_prefix}/{rep_name}_{pred_set}_preds_for_correlation_plot_model_5x128KB.npz",
                 true_labels=y_valid.cpu().numpy(),
                 cnn_predict=valid_preds.cpu().numpy(),
                 gc_preds=gc_preds.cpu().numpy(),
                 gc_preds_diff=gc_preds_diff.cpu().numpy()
                 )

        print(f" {pred_set} loss: {valid_loss:4.4f}", flush=True)
        print(f"Spearmanr: {spearmanr_twinc}," f"Pearsonr: {pearsonr_twinc}", flush=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file", type=str, default="config_data_classify.yml", help="path to the config file."
    )

    parser.add_argument(
        "--pred_set", type=str, choices=["validation", "test"], default="validation", help="Predict validation or test."
    )

    args = parser.parse_args()

    print(f"Predict {args.pred_set} set.")

    run_twinc_reg_inference(args.config_file, args.pred_set)


if __name__ == "__main__":
    main()
