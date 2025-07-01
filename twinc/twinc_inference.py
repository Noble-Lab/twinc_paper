"""
twinc_inference.py
Author: Anupama Jha <anupamaj@uw.edu>
"""

import torch
import pyfaidx
import argparse
import numpy as np
import configparser
from twinc_network import TwinCNet
from twinc_train import extract_set_data, TwinCDataGenerator
from twinc_utils import count_pos_neg, decode_chrome_order_dict, decode_list, gc_predictor
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve


def test_function(config_file,
                  partition):
    """
    Inference using TwinC network
    :param config_file: str, path to config file
    :param partition: str, val or test.
    :return: None
    """
    # read parameters from config file
    config = configparser.ConfigParser()
    config.read(config_file)

    chroms_order = decode_chrome_order_dict(
        config['data_parameters']['chroms_order'])
    print(f"chroms_order: {chroms_order}")

    # hg38 sequence file
    sequences_file = config['input_files']['seq_file']

    # read sequence fasta file
    sequences = pyfaidx.Fasta(sequences_file)
    all_chroms_keys = sorted(sequences.keys())
    print(f"all_chroms_keys: {all_chroms_keys}")

    # get chromosome lengths
    seq_chrom_lengths = []
    for chr_val in all_chroms_keys:
        len_value = len(sequences[chr_val][:].seq)
        print(f"{chr_val}, {len_value}")
        seq_chrom_lengths.append(len_value)

    # get chromosome names
    seq_chrom_names = []
    for chr_val in all_chroms_keys:
        if 'chr' in chr_val:
            seq_chrom_names.append(chr_val)
        else:
            seq_chrom_names.append(f"chr{chr_val}")

    # get chromosome start and end index
    # in the same order as the memory map
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

    label_key = f"labels_file_{partition}"
    # labels file
    labels_file = np.loadtxt(config['input_files'][label_key], delimiter="\t", dtype=str)

    save_prefix = config['output_files']['save_prefix']

    chrom_set = f"{partition}_chroms"
    # Get the list of chromosomes for validation
    set_chroms = decode_list(config['data_parameters'][chrom_set])
    print(f"labels_file: {labels_file}")
    set_loci = extract_set_data(labels_file, set_chroms, seq_chrom_start)
    print(f"set_chroms: {set_chroms}, set_loci: {len(set_loci)}")

    count_pos_neg(labels=np.array(set_loci[:, 6], dtype=int), set_name=f"{partition} set")

    # path to save the best model
    best_save_model = config["model_parameters"]["best_model_path"]
    # number of threads to process training data fetch
    num_workers = int(config["train_parameters"]["num_workers"])
    # batch size of training data
    batch_size = int(config["train_parameters"]["batch_size"])

    rep_name = config["data_parameters"]["rep_name"]

    set_gen = TwinCDataGenerator(seq_memory_map,
                                 set_loci,
                                 seq_chrom_start,
                                 seq_chrom_end,
                                 in_window=100000,
                                 reverse_complement=False,
                                 random_state=None
                                 )

    # Wrap it in a data loader
    set_gen = torch.utils.data.DataLoader(
        set_gen,
        pin_memory=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )
    # Move the model to the appropriate device
    model = TwinCNet()
    model.load_state_dict(torch.load(best_save_model))
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        y_valid = torch.empty((0, 2))
        valid_preds = torch.empty((0, 2)).to(device)
        gc_preds = torch.empty((0, 1)).to(device)
        gc_preds_diff = torch.empty((0, 1)).to(device)
        for cnt, data in enumerate(set_gen):
            if cnt % 1000 == 0:
                print(cnt)
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
            if cnt > 4999:
                break
        count_pos_neg(np.argmax(y_valid, axis=1), set_name="test set")
        valid_preds, y_valid = valid_preds.to(device), y_valid.to(device)
        # compute cross_entropy loss for the validation set.
        cross_entropy_loss = model.cross_entropy_loss(valid_preds, y_valid)

        # Extract the validation loss
        valid_loss = cross_entropy_loss.item()

        # Compute AUROC
        sklearn_rocauc = roc_auc_score(y_valid.cpu().numpy()[:, 1],
                                       valid_preds.cpu().numpy()[:, 1])

        # Compute AUPR/Average precision
        sklearn_ap = average_precision_score(y_valid.cpu().numpy()[:, 1],
                                             valid_preds.cpu().numpy()[:, 1])

        # Compute AUROC
        sklearn_rocauc_gc = roc_auc_score(y_valid.cpu().numpy()[:, 1],
                                          gc_preds.cpu().numpy()[:, 0])

        # Compute AUPR/Average precision
        sklearn_ap_gc = average_precision_score(y_valid.cpu().numpy()[:, 1],
                                                gc_preds.cpu().numpy()[:, 0])

        # Compute AUROC
        sklearn_rocauc_gc_diff = roc_auc_score(y_valid.cpu().numpy()[:, 1],
                                               gc_preds_diff.cpu().numpy()[:, 0])

        # Compute AUPR/Average precision
        sklearn_ap_gc_diff = average_precision_score(y_valid.cpu().numpy()[:, 1],
                                                     gc_preds_diff.cpu().numpy()[:, 0])

        np.savez(f"{save_prefix}/{rep_name}_preds_for_auroc_plot_model_{partition}.npz",
                 true_labels=y_valid.cpu().numpy()[:, 1],
                 cnn_predict=valid_preds.cpu().numpy()[:, 1],
                 gc_preds=gc_preds.cpu().numpy()[:, 0],
                 gc_preds_diff=gc_preds_diff.cpu().numpy()[:, 0]
                 )

        print(f" Test loss: {valid_loss:4.4f}")
        print(f"AUPR: {sklearn_ap}," f"AUROC: {sklearn_rocauc}")
        print(f"AUPR GC: {sklearn_ap_gc}," f"AUROC GC: {sklearn_rocauc_gc}")
        print(f"AUPR GC Diff: {sklearn_rocauc_gc_diff}," f"AUROC GC Diff: {sklearn_ap_gc_diff}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default="config_data_classify.yml",
        help="path to the config file."
    )

    parser.add_argument(
        "--partition",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Run validation set or test set."
    )

    args = parser.parse_args()

    print(f"Predict using TwinC network.")

    test_function(args.config_file, args.partition)


if __name__ == "__main__":
    main()
