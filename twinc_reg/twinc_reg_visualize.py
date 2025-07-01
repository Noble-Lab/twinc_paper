"""
twinc_reg_visualize.py
Author: Anupama Jha <anupamaj@uw.edu>
Code for visualizing the predictions from the
 TwinC regression model at chromosome level.
"""
import os
import gzip
import torch
import pyfaidx
import argparse
import numpy as np
import _pickle as pickle
import configparser
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from trans_contact_reg_data import TransContacts, GenomeWideTransContacts
from twinc_reg_network import TwinCRegNet
from trans_utils import count_pos_neg, decode_chrome_order_dict, decode_list, decode_chrome_order_inv_dict

plt.rcParams['axes.facecolor'] = 'gray'


class TwinCPredictDataGenerator(torch.utils.data.Dataset):
    """
    Data Generator for the predictions for TwinC regression.
    """

    def __init__(self,
                 seq_memmap,
                 set_loci,
                 seq_chrom_start,
                 seq_chrom_end,
                 reverse_complement=False,
                 in_window=640000,
                 random_state=None,
                 rev_comp_1=False,
                 rev_comp_2=False
                 ):
        """
        Given the sequence and genomic loci, generate requisite
        data for the model to make a prediction.
        :param seq_memmap: str, path to the one-hot-encoded sequence map.
        :param set_loci: np.array, set of genomic loci.
        :param seq_chrom_start: dict, chromosome start indexes in the sequence memory map.
        :param seq_chrom_end: dict, chromosome end indexes in the sequence memory map.
        :param reverse_complement: bool, reverse complement the sequence if needed.
        :param in_window: int, size of input window.
        :param random_state: int, define random state for deterministic behavior.
        :param rev_comp_1: bool, whether to reverse complement the first sequence in the input.
        :param rev_comp_2: bool, whether to reverse complement the second sequence in the input.
        """
        # Input sequence size
        self.in_window = in_window
        # If we want to reverse
        # complement the sequence
        self.reverse_complement = reverse_complement
        # Random state to use
        self.random_state = random_state

        self.reverse_comp_1 = rev_comp_1
        self.reverse_comp_2 = rev_comp_2

        self.seq_chrom_start = seq_chrom_start
        self.seq_chrom_end = seq_chrom_end

        # One hot encoded sequence
        # memory map.
        self.seq_memmap = seq_memmap
        self.chromA = set_loci[:, 0]
        self.chromB = set_loci[:, 3]
        print(f"self.chromA: {self.chromA}")
        print(f"self.chromB: {self.chromB}")
        # Start and end idx
        # of sequences in pos A
        self.chromA_coords = np.array(set_loci[:, 1:3], dtype=int)
        # Start and end idx
        # of sequences in pos B
        self.chromB_coords = np.array(set_loci[:, 4:6], dtype=int)
        # Hi-C contact values

    def __len__(self):
        # Return length of the data generator
        return len(self.chromA_coords)

    def __getitem__(self, i):
        """
        Extract batches of data from the data generator.
        :param i: int, index of element to extract.
        :return: two sequences (X1, X2) and labels (y)
        """
        # Initialize an all zero array for sequence A
        seq_array_A = np.zeros((4, self.in_window), dtype=np.float64)
        start_val_A = 0
        end_val_A = self.in_window

        # Get the memory map coordinates for sequence A
        s_idx_A, e_idx_A = self.chromA_coords[i][0], self.chromA_coords[i][1]

        if e_idx_A > self.seq_chrom_end[self.chromA[i]]:
            e_idx_A = self.seq_chrom_end[self.chromA[i]]

        if (e_idx_A - s_idx_A) < self.in_window:
            end_val_A = start_val_A + (e_idx_A - s_idx_A)
        elif (e_idx_A - s_idx_A) > self.in_window:
            e_idx_A = s_idx_A + self.in_window

        # Put relevant sequence A one-hot-encoding in the array
        seq_array_A[:, start_val_A:end_val_A] = self.seq_memmap[:, s_idx_A:e_idx_A]

        # Initialize an all zero array for sequence B
        seq_array_B = np.zeros((4, self.in_window), dtype=np.float64)

        start_val_B = 0
        end_val_B = self.in_window

        # Get the memory map coordinates for sequence B
        s_idx_B, e_idx_B = self.chromB_coords[i][0], self.chromB_coords[i][1]

        if e_idx_B > self.seq_chrom_end[self.chromB[i]]:
            e_idx_B = self.seq_chrom_end[self.chromB[i]]

        if (e_idx_B - s_idx_B) < self.in_window:
            end_val_B = start_val_B + (e_idx_B - s_idx_B)
        elif (e_idx_B - s_idx_B) > self.in_window:
            e_idx_B = s_idx_B + self.in_window

        # Put relevant sequence B one-hot-encoding in the array
        seq_array_B[:, start_val_B:end_val_B] = self.seq_memmap[:, s_idx_B:e_idx_B]

        # print(f"seq_array_A: {seq_array_A.shape}")
        # print(f"seq_array_B: {seq_array_B.shape}")

        # Convert arrays to float tensor
        X1 = torch.tensor(seq_array_A).float()
        X2 = torch.tensor(seq_array_B).float()

        if self.reverse_comp_1:
            X1 = torch.flip(X1, [1, 0])
        if self.reverse_comp_2:
            X2 = torch.flip(X2, [1, 0])

        return X1, X2, torch.tensor(i)


def extract_chrom_level_coords(labels_file,
                               chrom_starts,
                               seq_chrom_length):
    """

    :param labels_file:
    :param chrom_starts:
    :param seq_chrom_length:
    :return:
    """
    with gzip.open(labels_file, 'rb') as labels_obj:
        common_matrix = pickle.load(labels_obj)
        all_loci_pairs = []
        ij_idxs = []
        for i in range(common_matrix.hic_sum.shape[0]):
            for j in range(common_matrix.hic_sum.shape[1]):
                chrA = common_matrix.chromA
                chrB = common_matrix.chromB
                if 'chr' not in chrA:
                    chrA = f"chr{chrA}"
                if 'chr' not in chrB:
                    chrB = f"chr{chrB}"
                chrA_start = max(0, (i - 2) * common_matrix.resolution) + chrom_starts[chrA]
                chrA_end = min(seq_chrom_length[chrA], (i + 3) * common_matrix.resolution) + chrom_starts[chrA]
                chrB_start = max(0, (j - 2) * common_matrix.resolution) + chrom_starts[chrB]
                chrB_end = min(seq_chrom_length[chrB], (j + 3) * common_matrix.resolution) + chrom_starts[chrB]
                all_loci_pairs.append([chrA, chrA_start, chrA_end,
                                       chrB, chrB_start, chrB_end])
                ij_idxs.append([i - 2, i - 1, i, i + 1, i + 2, j - 2, j - 1, j, j + 1, j + 2])
                # print([i-2, i-1, i, i+1, i+2, j-2, j-1, j, j+1, j+2])
        print(f"chrom_starts[chrA]: {chrom_starts[chrA]}", flush=True)
        print(f"chrom_starts[chrB]: {chrom_starts[chrB]}", flush=True)
        print(f"all_loci_pairs[0]: {all_loci_pairs[0]}", flush=True)
        print(f"all_loci_pairs[-1]: {all_loci_pairs[-1]}", flush=True)
        print(f"ij_idxs[0]: {ij_idxs[0]}", flush=True)
        print(f"ij_idxs[-1]: {ij_idxs[-1]}", flush=True)
        all_loci_pairs = np.array(all_loci_pairs, dtype=str)
        ij_idxs = np.array(ij_idxs, dtype=int)
        return all_loci_pairs, ij_idxs


def genome_wide_predict_pipeline(config_file="config_data_classify.yml",
                                 pred_set="validation",
                                 rev_comp_1="False",
                                 rev_comp_2="False",
                                 save_suffix="_pos_strand"):
    """
    Make predictions for every trans-chromosome pair.
    :param config_file: str, path to the config file
    :param pred_set: list, validation or test
    :param rev_comp_1: bool, whether to reverse complement the first sequence in the input.
    :param rev_comp_2: bool, whether to reverse complement the second sequence in the input.
    :param save_suffix: str, path prefix to output files, make sure folder exists.
    :return: None.
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    # chromosome order
    chroms_order = decode_chrome_order_inv_dict(
        config['data_parameters']['chroms_order_inv'])
    print(f"chroms_order: {chroms_order}", flush=True)

    selected_resolution = int(config['data_parameters']['selected_resolution'])
    save_prefix = config['output_files']['save_prefix']

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
    seq_chrom_length = {}
    cum_total = 0
    for i in range(len(all_chroms_keys)):
        seq_chrom_start[seq_chrom_names[i]] = cum_total
        cum_total += seq_chrom_lengths[i]
        seq_chrom_end[seq_chrom_names[i]] = cum_total - 1
        seq_chrom_length[seq_chrom_names[i]] = seq_chrom_lengths[i]

    memmap_shape = (4, int(config["data_parameters"]["memmap_length"]))
    device = config["train_parameters"]["device"]

    hg38_memmap_path = config["input_files"]["seq_memmap"]

    # Load the genome into memory
    seq_memory_map = np.memmap(hg38_memmap_path,
                               dtype="float32",
                               mode="r",
                               shape=memmap_shape)

    if pred_set == "validation":
        # Get the list of chromosomes for validation
        test_chroms = decode_list(config['data_parameters']['val_chroms'])
        print("val_chroms: ", test_chroms, len(chroms_order), flush=True)

    elif pred_set == "test":
        # Get the list of chromosomes for validation
        test_chroms = decode_list(config['data_parameters']['test_chroms'])
        print("test_chroms: ", test_chroms, len(chroms_order), flush=True)

    rep_name = config["data_parameters"]["rep_name"]

    # path to save best model
    best_save_model = config["model_parameters"]["best_model_path"]

    final_save_model = config["model_parameters"]["final_model_path"]
    # number of threads to process training data fetch
    num_workers = int(config["train_parameters"]["num_workers"])
    # batch size of training data
    batch_size = int(config["train_parameters"]["batch_size"])
    for i in range(1, len(chroms_order) + 1, 1):
        for j in range(i + 1, len(chroms_order) + 1, 1):
            # labels file
            chromA = chroms_order[i]
            chromB = chroms_order[j]

            if 'chr' not in chromA:
                chromA = f"chr{chromA}"

            if 'chr' not in chromB:
                chromB = f"chr{chromB}"
            # If both chromosomes are in the set
            if chromA in test_chroms and chromB in test_chroms:
                save_test_preds = f"{save_prefix}/{chromA}_{chromB}_test_preds_{save_suffix}_5x128KB_tmp.npz"

                print(f"{chromA} {chromB}", flush=True)
                labels_file = f"{save_prefix}/{chromA}_{chromB}_{rep_name}_{int(selected_resolution / 5.0)}.gz"
                if not os.path.exists(labels_file):
                    chromA_nochr = chromA.replace("chr", "")
                    chromB_nochr = chromB.replace("chr", "")
                    labels_file = f"{save_prefix}/{chromA_nochr}_{chromB_nochr}_{rep_name}_" \
                                  f"{int(selected_resolution / 5.0)}.gz"

                print(f"{chromA} {chromB}", flush=True)
                test_loci, test_matrix_idxs = extract_chrom_level_coords(labels_file,
                                                                         seq_chrom_start,
                                                                         seq_chrom_length)
                print(f"test_loci: {len(test_loci)}", flush=True)
                print(f"test_matrix_idxs: {len(test_matrix_idxs)}", flush=True)

                if not os.path.exists(save_test_preds):
                    # Select correct reverse complement option.
                    if rev_comp_1 == "True" and rev_comp_2 == "True":
                        test_gen = TwinCPredictDataGenerator(seq_memory_map,
                                                             test_loci,
                                                             seq_chrom_start,
                                                             seq_chrom_end,
                                                             in_window=selected_resolution,
                                                             reverse_complement=False,
                                                             random_state=None,
                                                             rev_comp_1=True,
                                                             rev_comp_2=True)
                    if rev_comp_1 == "True" and rev_comp_2 == "False":
                        test_gen = TwinCPredictDataGenerator(seq_memory_map,
                                                             test_loci,
                                                             seq_chrom_start,
                                                             seq_chrom_end,
                                                             in_window=selected_resolution,
                                                             reverse_complement=False,
                                                             random_state=None,
                                                             rev_comp_1=True,
                                                             rev_comp_2=False)
                    if rev_comp_1 == "False" and rev_comp_2 == "True":
                        test_gen = TwinCPredictDataGenerator(seq_memory_map,
                                                             test_loci,
                                                             seq_chrom_start,
                                                             seq_chrom_end,
                                                             in_window=selected_resolution,
                                                             reverse_complement=False,
                                                             random_state=None,
                                                             rev_comp_1=False,
                                                             rev_comp_2=True)
                    if rev_comp_1 == "False" and rev_comp_2 == "False":
                        test_gen = TwinCPredictDataGenerator(seq_memory_map,
                                                             test_loci,
                                                             seq_chrom_start,
                                                             seq_chrom_end,
                                                             in_window=selected_resolution,
                                                             reverse_complement=False,
                                                             random_state=None,
                                                             rev_comp_1=False,
                                                             rev_comp_2=False)

                    # Wrap it in a data loader
                    test_gen = torch.utils.data.DataLoader(
                        test_gen,
                        pin_memory=True,
                        num_workers=32,
                        batch_size=32,
                        shuffle=False,
                    )

                    # Move the model to appropriate device
                    model = TwinCRegNet()
                    model.load_state_dict(torch.load(best_save_model))
                    # model.load_state_dict(torch.load(final_save_model))
                    model = model.to(device)
                    with torch.no_grad():
                        model.eval()
                        test_preds = torch.empty((0, 5, 5)).to(device)
                        all_chrom_pos = []
                        for cnt, data in enumerate(test_gen):
                            if cnt % 500 == 0:
                                print(cnt, flush=True)
                            # Get features
                            X1, X2, index = data
                            all_chrom_pos.extend(index.numpy())

                            # Convert them to float
                            X1, X2 = X1.float(), X2.float()
                            X1, X2 = X1.to(device), X2.to(device)

                            # Run forward pass
                            test_pred = model.forward(X1, X2)
                            # print(f"test_pred: {test_pred}")
                            test_preds = torch.cat((test_preds, test_pred))
                            test_preds = test_preds.to(device)

                        test_preds = test_preds.cpu().numpy()
                        print(f"test_preds: {test_preds.shape}", flush=True)
                        # save predictions
                        save_test_preds = f"{save_prefix}/{chromA}_{chromB}_test_preds_{save_suffix}_5x128KB_tmp.npz"

                        np.savez(save_test_preds, test_preds=test_preds)

                else:
                    test_preds = np.load(save_test_preds)["test_preds"]

                with gzip.open(labels_file, 'rb') as labels_obj:
                    common_matrix = pickle.load(labels_obj)
                    true_matrix = common_matrix.hic_sum_noartifacts
                    pred_matrix = np.zeros(common_matrix.hic_sum_noartifacts.shape)
                    count_matrix = np.zeros(common_matrix.hic_sum_noartifacts.shape)
                    for ii in range(len(test_preds)):

                        i_idx = test_matrix_idxs[ii][2]
                        j_idx = test_matrix_idxs[ii][7]

                        if 1 < i_idx < pred_matrix.shape[0] - 2:
                            if 1 < j_idx < pred_matrix.shape[1] - 2:
                                pred_matrix[i_idx - 2:i_idx + 3, j_idx - 2:j_idx + 3] += test_preds[ii, 0:5, 0:5]
                                count_matrix[i_idx - 2:i_idx + 3, j_idx - 2:j_idx + 3] += 1

                    print(f"How many zeros in count_matrix: {np.count_nonzero(count_matrix == 0)}")
                    print(f"count_matrix: max: {np.max(count_matrix)}, min: {np.min(count_matrix)}")
                    print(
                        f"pred_matrix: min {np.nanmin(pred_matrix)}, "
                        f"max {np.nanmax(pred_matrix)}, "
                        f"mean {np.nanmean(pred_matrix)}")
                    print(
                        f"pred_matrix: min {np.nanmin(pred_matrix)}, "
                        f"max {np.nanmax(pred_matrix)}, "
                        f"mean {np.nanmean(pred_matrix)}")

                    print(f"pred_matrix: {pred_matrix}")

                    np.savez(f"{save_prefix}/{chromA}_{chromB}_test_results_reg_{save_suffix}_5x128KB_tmp.npz",
                             hic_sum=common_matrix.hic_sum,
                             hic_sum_noartifacts=common_matrix.hic_sum_noartifacts,
                             hic_pos_num_sup=common_matrix.hic_pos_num_sup,
                             hic_neg_num_sup=common_matrix.hic_neg_num_sup,
                             true_matrix=true_matrix,
                             pred_matrix=pred_matrix,
                             count_matrix=count_matrix)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file", type=str, default="config_data_classify.yml", help="path to the config file."
    )

    parser.add_argument(
        "--pred_set", type=str, choices=["validation", "test"], default="validation", help="Predict validation or test."
    )

    parser.add_argument(
        "--rev_comp_1", type=str, choices=["True", "False"], default="False", help="Reverse Complement seq 1"
    )

    parser.add_argument(
        "--rev_comp_2", type=str, choices=["True", "False"], default="False", help="Reverse Complement seq 2"
    )

    parser.add_argument(
        "--save_suffix", type=str, help="suffix for the save file"
    )

    args = parser.parse_args()

    print(f"Predict whole chromosome {args.pred_set} set.")
    genome_wide_predict_pipeline(config_file=args.config_file,
                                 pred_set=args.pred_set,
                                 rev_comp_1=args.rev_comp_1,
                                 rev_comp_2=args.rev_comp_2,
                                 save_suffix=args.save_suffix)


if __name__ == "__main__":
    main()
