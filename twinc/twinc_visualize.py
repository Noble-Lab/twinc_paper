"""
twinc_visualize.py
Author: Anupama Jha <anupamaj@uw.edu>
TwinC classification chromosome-wide prediction for visualization purposes.
"""
import os
import gzip
import torch
import pyfaidx
import hicstraw
import argparse
import numpy as np
import configparser
import seaborn as sns
import _pickle as pickle
import matplotlib.pyplot as plt
from twinc_network import TwinCNet
from trans_contact_data import TransContacts
from twinc_utils import gc_predictor, decode_chrome_order_inv_dict, decode_list

plt.rcParams['axes.facecolor'] = 'gray'


class TwinCPredictGenerator(torch.utils.data.Dataset):
    """
    TwinC predictor class
    """

    def __init__(self,
                 seq_memmap,
                 set_loci,
                 reverse_complement=False,
                 in_window=100000,
                 random_state=None
                 ):
        """
        Contructor that takes the sequence memory map
        and locus pairs to produce predictions.
        :param seq_memmap: str, path to one-hot-encoded genomic sequence memory map.
        :param set_loci: np.array, genomic loci for which prediction is to be made.
        :param reverse_complement: bool, do we need to reverse complement the sequence?
        :param in_window: int, size of window
        :param random_state: int, random state for deterministic behavior.
        """
        # Input sequence size
        self.in_window = in_window
        # If we want to reverse
        # complement the sequence
        self.reverse_complement = reverse_complement
        # Random state to use
        self.random_state = random_state

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

        # Put relevant sequence A one-hot-encoding in the array
        seq_array_A[:, start_val_A:end_val_A] = self.seq_memmap[:, s_idx_A:e_idx_A]

        # Initialize an all zero array for sequence B
        seq_array_B = np.zeros((4, self.in_window), dtype=np.float64)

        start_val_B = 0
        end_val_B = self.in_window

        # Get the memory map coordinates for sequence B
        s_idx_B, e_idx_B = self.chromB_coords[i][0], self.chromB_coords[i][1]

        # Put relevant sequence B one-hot-encoding in the array
        seq_array_B[:, start_val_B:end_val_B] = self.seq_memmap[:, s_idx_B:e_idx_B]

        # print(f"seq_array_A: {seq_array_A.shape}")
        # print(f"seq_array_B: {seq_array_B.shape}")

        # Convert arrays to float tensor
        X1 = torch.tensor(seq_array_A).float()
        X2 = torch.tensor(seq_array_B).float()

        return X1, X2, torch.tensor(i)


def extract_chrom_level_coords(labels_file,
                               chrom_starts):
    """
    Get genomic coordinates for chromosome pairs using the
    chromosome starts in the memory.
    :param labels_file: str, labels file.
    :param chrom_starts: dict, keys are chromosomes,
                               values are start coordinates.
    :return: np.array, np.array. All locus pairs and corresponding indexes
            in the chromosome pair matrix for visualization.
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
                chrA_start = i * common_matrix.resolution + chrom_starts[chrA]
                chrA_end = (i + 1) * common_matrix.resolution + chrom_starts[chrA]
                chrB_start = j * common_matrix.resolution + chrom_starts[chrB]
                chrB_end = (j + 1) * common_matrix.resolution + chrom_starts[chrB]
                all_loci_pairs.append([chrA, chrA_start, chrA_end,
                                       chrB, chrB_start, chrB_end])
                ij_idxs.append([i, j])
        print(f"chrom_starts[chrA]: {chrom_starts[chrA]}")
        print(f"chrom_starts[chrB]: {chrom_starts[chrB]}")
        print(f"all_loci_pairs[0]: {all_loci_pairs[0]}")
        print(f"all_loci_pairs[-1]: {all_loci_pairs[-1]}")
        print(f"ij_idxs[0]: {ij_idxs[0]}")
        print(f"ij_idxs[-1]: {ij_idxs[-1]}")
        all_loci_pairs = np.array(all_loci_pairs, dtype=str)
        ij_idxs = np.array(ij_idxs, dtype=int)
        return all_loci_pairs, ij_idxs


def genome_wide_predict_pipeline(config_file="config_data_classify.yml"):
    """
    Predict the chromosome pairs in a set.
    :param config_file: str, config file
    :return: None.
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    # chromosome order
    chroms_order = decode_chrome_order_inv_dict(
        config['data_parameters']['chroms_order_inv'])
    print(f"chroms_order: {chroms_order}")

    selected_resolution = int(config['data_parameters']['selected_resolution'])
    save_prefix = config['output_files']['save_prefix']

    # hg38 sequence file
    sequences_file = config['input_files']['seq_file']

    # read sequence fasta file
    sequences = pyfaidx.Fasta(sequences_file)
    all_chroms_keys = sorted(sequences.keys())
    print(f"all_chroms_keys: {all_chroms_keys}")

    # get chromsome lengths
    seq_chrom_lengths = []
    for chr_val in all_chroms_keys:
        len_value = len(sequences[chr_val][:].seq)
        print(f"{chr_val}, {len_value}")
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

    # Get the list of chromosomes for validation
    test_chroms = decode_list(config['data_parameters']['test_chroms'])
    print("test_chroms: ", test_chroms, len(chroms_order))

    rep_name = config["data_parameters"]["rep_name"]

    # path to save best model
    best_save_model = config["model_parameters"]["best_model_path"]
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

            if chromA in test_chroms and chromB in test_chroms:
                print(f"{chromA} {chromB}")
                labels_file = f"{save_prefix}/{chromA}_{chromB}_{rep_name}_100000_5_v_5.gz"
                if not os.path.exists(labels_file):
                    chromA_nochr = chromA.replace("chr", "")
                    chromB_nochr = chromB.replace("chr", "")
                    labels_file = f"{save_prefix}/{chromA_nochr}_{chromB_nochr}_{rep_name}_100000_5_v_5.gz"

                print(f"{chromA} {chromB}")
                test_loci, test_matrix_idxs = extract_chrom_level_coords(labels_file, seq_chrom_start)
                print(f"test_loci: {len(test_loci)}")
                print(f"test_matrix_idxs: {len(test_matrix_idxs)}")
                test_gen = TwinCPredictGenerator(seq_memory_map,
                                                 test_loci,
                                                 in_window=100000,
                                                 reverse_complement=False,
                                                 random_state=None)

                # Wrap it in a data loader
                test_gen = torch.utils.data.DataLoader(
                    test_gen,
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
                    test_preds = torch.empty((0, 2)).to(device)
                    gc_preds = torch.empty((0, 1)).to(device)
                    gc_preds_diff = torch.empty((0, 1)).to(device)
                    all_chrom_pos = []
                    for cnt, data in enumerate(test_gen):
                        if cnt % 1000 == 0:
                            print(cnt, flush=True)
                        # if cnt > 2000:
                        #    break
                        # Get features
                        X1, X2, index = data
                        all_chrom_pos.extend(index.numpy())

                        # Convert them to float
                        X1, X2 = X1.float(), X2.float()
                        X1, X2 = X1.to(device), X2.to(device)

                        gc_pred, gc_pred_diff = gc_predictor(X1, X2, is_torch=True)
                        gc_preds = torch.cat((gc_preds, gc_pred))
                        gc_preds = gc_preds.to(device)

                        gc_preds_diff = torch.cat((gc_preds_diff, gc_pred_diff))
                        gc_preds_diff = gc_preds_diff.to(device)

                        # Run forward pass
                        test_pred = model.forward(X1, X2)
                        # print(f"test_pred: {test_pred}")
                        test_preds = torch.cat((test_preds, test_pred))
                        test_preds = test_preds.to(device)

                    test_preds = test_preds.cpu().numpy()
                    print(f"test_preds: {test_preds.shape}")

                    gc_preds = gc_preds.cpu().numpy()
                    print(f"gc_preds: {gc_preds.shape}")

                    gc_preds_diff = gc_preds_diff.cpu().numpy()
                    print(f"gc_preds_diff: {gc_preds_diff.shape}")

                    save_test_preds = f"{save_prefix}/{chromA}_{chromB}_test_preds.npz"

                    np.savez(save_test_preds,
                             test_preds=test_preds,
                             gc_preds=gc_preds,
                             gc_preds_diff=gc_preds_diff)

                with gzip.open(labels_file, 'rb') as labels_obj:
                    common_matrix = pickle.load(labels_obj)
                    pred_matrix_pos = np.zeros((common_matrix.hic_sum_noartifacts.shape))
                    pred_matrix_neg = np.zeros((common_matrix.hic_sum_noartifacts.shape))
                    true_matrix = common_matrix.hic_pos_num_sup + -1 * common_matrix.hic_neg_num_sup
                    pred_matrix = np.zeros((common_matrix.hic_sum_noartifacts.shape))
                    pred_matrix_gc = np.zeros((common_matrix.hic_sum_noartifacts.shape))

                    print(f"pred_matrix.shape: {pred_matrix.shape}")
                    print(f"pred_matrix_gc.shape: {pred_matrix_gc.shape}")
                    for ii in range(len(test_preds)):

                        # print(f"test_matrix_idxs[ii]: {test_matrix_idxs[ii]}, {test_preds[ii][1]}")
                        pred_matrix_pos[test_matrix_idxs[ii][0], test_matrix_idxs[ii][1]] = test_preds[ii][1]
                        pred_matrix_neg[test_matrix_idxs[ii][0], test_matrix_idxs[ii][1]] = test_preds[ii][0]
                        pred_matrix_gc[test_matrix_idxs[ii][0], test_matrix_idxs[ii][1]] = gc_preds[ii][0]

                        if np.argmax(test_preds[ii]) == 0:
                            pred_matrix[test_matrix_idxs[ii][0], test_matrix_idxs[ii][1]] = -1  # *test_preds[idx][0]
                        else:
                            pred_matrix[test_matrix_idxs[ii][0], test_matrix_idxs[ii][1]] = 1  # test_preds[idx][1]

                    print(f"{save_prefix}/{chromA}_{chromB}_test_results.npz")
                    np.savez(f"{save_prefix}/{chromA}_{chromB}_test_results.npz",
                             hic_sum=common_matrix.hic_sum,
                             hic_sum_noartifacts=common_matrix.hic_sum_noartifacts,
                             hic_pos_num_sup=common_matrix.hic_pos_num_sup,
                             hic_neg_num_sup=common_matrix.hic_neg_num_sup,
                             pred_matrix_pos=pred_matrix_pos,
                             pred_matrix_neg=pred_matrix_neg,
                             true_matrix=true_matrix,
                             pred_matrix=pred_matrix,
                             pred_matrix_gc=pred_matrix_gc)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file", type=str, default="config_data_classify.yml", help="path to the config file."
    )

    args = parser.parse_args()
    genome_wide_predict_pipeline(config_file=args.config_file)


if __name__ == "__main__":
    main()
