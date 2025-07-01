"""
twinc_reg_train.py
Author: Anupama Jha <anupamaj@uw.edu>
Training code for TwinC regression model
"""
import os
import torch
import pyfaidx
import argparse
import numpy as np
import configparser
from scipy.stats import pearsonr, spearmanr
from twinc_reg_network import TwinCRegNet
from trans_utils import count_pos_neg, decode_chrome_order_dict, decode_list


def extract_set_data(labels_file,
                     set_chrs,
                     chrom_starts,
                     mean_expectation,
                     eps,
                     shuffle=False):
    """
    Extract the training/val labels from the labels file.
    :param labels_file: np.array, labels with genomic coordinates
    :param set_chrs: list, list of chromosomes we want labels from.
    :param chrom_starts: dict, keys are chromosome names, values are
                               start indexes in the sequence memory map.
    :param mean_expectation: float, mean expectation
    :param eps: float, error tolerance
    :param shuffle: bool, whether to shuffle data or not
    :return: np.array, np.array, genomic loci, regression labels
    """
    set_loci = []
    reg_targets = []
    for i in range(len(labels_file)):
        if i % 1000000 == 0:
            print(f"{i} of {len(labels_file)} processed", flush=True)
        labels_line = labels_file[i].strip("\n").split("\t")
        chrA = labels_line[0]
        chrB = labels_line[3]
        if 'chr' not in chrA:
            chrA = f"chr{chrA}"
        if 'chr' not in chrB:
            chrB = f"chr{chrB}"
        if chrA in set_chrs and chrB in set_chrs:
            chrA_start = max(0, int(labels_line[1])) + chrom_starts[chrA]
            chrA_end = max(0, int(labels_line[2])) + chrom_starts[chrA]
            chrB_start = max(0, int(labels_line[4])) + chrom_starts[chrB]
            chrB_end = max(0, int(labels_line[5])) + chrom_starts[chrB]
            label = int(labels_line[6])
            reg_target = labels_line[7].split(",")
            # print(f"reg_target: {len(reg_target)}")
            if len(reg_target) == 25:
                reg_target = np.array(reg_target, dtype=float)
                if np.sum(np.isnan(reg_target)) <= 5:
                    # print(f"reg_target: {reg_target}")
                    reg_target = np.log((reg_target + eps) / (mean_expectation + eps))
                    set_loci.append([chrA, chrA_start, chrA_end,
                                     chrB, chrB_start, chrB_end,
                                     label])
                    reg_targets.append(reg_target)

    set_loci = np.array(set_loci, dtype=str)
    reg_targets = np.array(reg_targets, dtype=float)
    reg_targets = reg_targets.reshape((reg_targets.shape[0], 5, 5))
    print(f"before set_loci: {set_loci.shape}", flush=True)
    print(f"reg_targets: {reg_targets.shape}", flush=True)

    pos = np.where(np.array(set_loci[:, 6], dtype=int) == 1)[0]
    neg = np.where(np.array(set_loci[:, 6], dtype=int) == 0)[0]

    print(f"{len(pos)}" f" positives and {len(neg)} negatives, total {len(set_loci)}", flush=True)

    return set_loci, reg_targets


def run_twinc_reg_training(config_file):
    """
    Run TwinC regression training
    :param config_file: str, path to the config file
    :return: None
    """
    # read parameters from config file
    config = configparser.ConfigParser()
    config.read(config_file)

    mean_expectation = float(config['data_parameters']['mean_expectation'])
    eps = float(config['data_parameters']['eps'])

    selected_resolution = int(config['data_parameters']['selected_resolution'])
    chroms_order = decode_chrome_order_dict(
        config['data_parameters']['chroms_order'])
    print(f"chroms_order: {chroms_order}", flush=True)

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
    print(f"memmap_shape: {memmap_shape}", flush=True)
    device = config["train_parameters"]["device"]

    hg38_memmap_path = config["input_files"]["seq_memmap"]

    # Load the genome into memory
    seq_memory_map = np.memmap(hg38_memmap_path,
                               dtype="float32",
                               mode="r",
                               shape=memmap_shape)

    # labels file
    # labels_file_train = np.loadtxt(config['input_files']['labels_file_train'], delimiter="\t", dtype=str)
    labels_file_train = open(config['input_files']['labels_file_train'], 'r').readlines()

    # get the list of chromosomes for training
    train_chroms = decode_list(config['data_parameters']['train_chroms'])
    train_loci, train_reg_targets = extract_set_data(labels_file_train, train_chroms, seq_chrom_start, mean_expectation,
                                                     eps)
    print("train_chroms: ", train_chroms, flush=True)
    print(f"train_loci: {train_loci[0]}, {len(train_loci)}", flush=True)

    count_pos_neg(labels=np.array(train_loci[:, 6], dtype=int), set_name="Train set")

    # labels file
    labels_file_val = open(config['input_files']['labels_file_val'], 'r').readlines()

    # Get the list of chromosomes for validation
    val_chroms = decode_list(config['data_parameters']['val_chroms'])
    val_loci, val_reg_targets = extract_set_data(labels_file_val, val_chroms, seq_chrom_start, mean_expectation, eps,
                                                 shuffle=True)
    print("val_chroms: ", val_chroms, flush=True)

    count_pos_neg(labels=np.array(val_loci[:, 6], dtype=int), set_name="Val set")

    # path to save best model
    best_save_model = config["model_parameters"]["best_model_path"]
    # path to save final model
    final_save_model = config["model_parameters"]["final_model_path"]
    # maximum number of epochs for training
    max_epochs = int(config["train_parameters"]["max_epochs"])
    # number of threads to process training data fetch
    num_workers = int(config["train_parameters"]["num_workers"])
    # batch size of training data
    batch_size = int(config["train_parameters"]["batch_size"])
    # learning rate
    sup_lr = float(config["train_parameters"]["lr"])

    train_gen = TransHiCDataGenerator(seq_memory_map,
                                      train_loci,
                                      train_reg_targets,
                                      seq_chrom_start,
                                      seq_chrom_end,
                                      in_window=selected_resolution,
                                      reverse_complement=True,
                                      random_state=None
                                      )

    # Wrap it in a data loader
    train_gen = torch.utils.data.DataLoader(
        train_gen,
        pin_memory=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )

    val_gen = TransHiCDataGenerator(seq_memory_map,
                                    val_loci,
                                    val_reg_targets,
                                    seq_chrom_start,
                                    seq_chrom_end,
                                    in_window=selected_resolution,
                                    reverse_complement=True,
                                    random_state=None
                                    )

    # Wrap it in a data loader
    val_gen = torch.utils.data.DataLoader(
        val_gen,
        pin_memory=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )
    if not os.path.exists(best_save_model):
        print(f"No prior model, random initialization", flush=True)
        # Move the model to appropriate device
        model = TwinCRegNet().to(device)
    else:
        print(f"Loading prior model {best_save_model}", flush=True)
        model = TwinCRegNet()
        model.load_state_dict(torch.load(best_save_model))
        model = model.to(device)

    # Adam optimizer with learning
    # rate 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=sup_lr)
    validation_iter = 200  # int(np.floor(len(train_gen) / (batch_size * 10.0)))

    # Train the model
    model.fit_supervised(
        train_gen,
        optimizer,
        val_gen,
        max_epochs=10,
        validation_iter=validation_iter,
        device=device,
        best_save_model=best_save_model,
        final_save_model=final_save_model
    )


class TwinCDataGenerator(torch.utils.data.Dataset):
    def __init__(self,
                 seq_memmap,
                 set_loci,
                 seq_reg_targets,
                 seq_chrom_start,
                 seq_chrom_end,
                 in_window=640000,
                 reverse_complement=True,
                 random_state=None
                 ):
        """
        PyTorch Dataset which extracts two
        genome sequences from different
        chromosomes using a sequence
        memory map.
        :param seq_memmap: np.memmap, numpy
                        memory map storing
                        the genome
        :param trans_hic_data: SiameseHiCModelData,
                            object from data sampler
                            class containing the
                            trans-contact genomic
                            coordinate, Hi-C patch
                            and memory map coordinates
                            for the sequence to extract.
        :param in_window: int, size of one input sequence
                            total input = 2*in_window,
                            as we have two sequences.
        :param reverse_complement: Bool ,do we train
                                the model using
                                reverse complement
                                of the positive strand.

        :param random_state: np.random, seed for
                                the numpy random state.
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

        self.seq_chrom_start = seq_chrom_start
        self.seq_chrom_end = seq_chrom_end
        # Start and end idx
        # of sequences in pos A
        self.chromA_coords = np.array(set_loci[:, 1:3], dtype=int)
        # Start and end idx
        # of sequences in pos B
        self.chromB_coords = np.array(set_loci[:, 4:6], dtype=int)
        # Hi-C contact values

        self.labels = torch.tensor(np.array(seq_reg_targets, dtype=float))
        print(f"self.labels : {self.labels.shape}")

    def __len__(self):
        # Return length of the data generator
        return len(self.labels)

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

        # Convert arrays to float tensor
        X1 = torch.tensor(seq_array_A).float()
        X2 = torch.tensor(seq_array_B).float()
        y = self.labels[i].float()

        # If reverse complement is true
        if self.reverse_complement:
            # flip a coin, if it is >= 0.5
            if np.random.uniform(low=0, high=1, size=1)[0] >= 0.5:
                # flip X1
                X1 = torch.flip(X1, [1, 0])
                y = torch.flip(y, [0, ])
            # flip again, if it is >= 0.5
            if np.random.uniform(low=0, high=1, size=1)[0] >= 0.5:
                # flip X2
                X2 = torch.flip(X2, [1, 0])
                y = torch.flip(y, [1, ])

        # flip a coin, if its is >= 0.5
        if np.random.uniform(low=0, high=1, size=1)[0] >= 0.5:
            # switch X1 and X2
            tmp = X1
            X1 = X2
            X2 = tmp
            y = y.T

        return X1, X2, y


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        help="path to the config file."
    )

    args = parser.parse_args()

    print(f"Training a supervised CNN regression model.")

    run_twinc_reg_training(args.config_file)


if __name__ == "__main__":
    main()
