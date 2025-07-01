"""
twinc_reg_data.py
Author: Anupama Jha <anupamaj@uw.edu>
This script produces training,validation
and test data for training the twinc model.
This version takes mcool file as input.
Adaptive coarse-graining adapted from
Orca (https://github.com/jzhoulab/orca_manuscript).
"""
import os
import gzip
import cooler
import argparse
import configparser
import numpy as np
import _pickle as pickle
from trans_utils import decode_chrome_order_inv_dict, decode_list
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from cooltools.lib.numutils import adaptive_coarsegrain
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve


def _adaptive_coarsegrain(ar,
                          countar,
                          max_levels=12,
                          cuda=False):
    """
    Wrapper for cooltools adaptive coarse-graining to add support
    for non-square input for inter-chromosomal predictions.
    :param ar: np.array, array for adaptive coarse-graining.
    :param countar: int, counter for adaptive coarse-graining.
    :param max_levels: int, number of levels for adaptive coarse-graining.
    :param cuda: str, whether to use GPU
    :return: np.array, coarse-grained array
    """
    global adaptive_coarsegrain_fn
    if cuda:
        adaptive_coarsegrain_fn = adaptive_coarsegrain_gpu
    else:
        adaptive_coarsegrain_fn = adaptive_coarsegrain

    assert np.all(ar.shape == countar.shape)
    if ar.shape[0] < 9 and ar.shape[1] < 9:
        ar_padded = np.empty((9, 9))
        ar_padded.fill(np.nan)
        ar_padded[: ar.shape[0], : ar.shape[1]] = ar

        countar_padded = np.empty((9, 9))
        countar_padded.fill(np.nan)
        countar_padded[: countar.shape[0], : countar.shape[1]] = countar
        return adaptive_coarsegrain_fn(ar_padded, countar_padded, max_levels=max_levels)[
               : ar.shape[0], : ar.shape[1]
               ]

    if ar.shape[0] == ar.shape[1]:
        return adaptive_coarsegrain_fn(ar, countar, cutoff=5, max_levels=max_levels)
    elif ar.shape[0] > ar.shape[1]:
        padding = np.empty((ar.shape[0], ar.shape[0] - ar.shape[1]))
        padding.fill(np.nan)
        return adaptive_coarsegrain_fn(
            np.hstack([ar, padding]), np.hstack([countar, padding]), max_levels=max_levels
        )[:, : ar.shape[1]]
    elif ar.shape[0] < ar.shape[1]:
        padding = np.empty((ar.shape[1] - ar.shape[0], ar.shape[1]))
        padding.fill(np.nan)
        return adaptive_coarsegrain_fn(
            np.vstack([ar, padding]), np.vstack([countar, padding]), max_levels=max_levels
        )[: ar.shape[0], :]


def normalize_hic_data(genome_hic_cool,
                       fetch_str,
                       region_2,
                       resolution,
                       kernel_stddev=0.0):
    """
    Normalize and optionally plot normalized HiC data
    :param genome_hic_file: str, cool file with HiC data
    :param fetch_str: str, chromosome X coords to process from the HiC data
    :param region_2: str, chromosome Y coords to process from the HiC data
    :return:
    """

    if kernel_stddev > 0:
        kernel = Gaussian2DKernel(x_stddev=kernel_stddev)
    else:
        kernel = None
    seq_hic_raw = genome_hic_cool.matrix(balance=False).fetch(fetch_str, region_2)

    # Get KR normalized observed values from cool file
    seq_hic_bal = genome_hic_cool.matrix(balance=True).fetch(fetch_str, region_2)

    seq_hic_smoothed = _adaptive_coarsegrain(seq_hic_bal,
                                             seq_hic_raw,
                                             max_levels=12)

    if kernel is not None:
        seq_hic_smoothed_nans = np.isnan(seq_hic_smoothed)
        seq_hic_smoothed[seq_hic_smoothed_nans] = 0
        seq_hic_smoothed = convolve(seq_hic_smoothed, kernel)
        seq_hic_smoothed[seq_hic_smoothed_nans] = np.nan

    return seq_hic_smoothed


class TransContacts():
    def __init__(self,
                 chromA,
                 chromB,
                 chromA_len,
                 chromB_len,
                 resolution,
                 rep_name,
                 hic_files,
                 reads_threshold_pos,
                 reads_threshold_neg
                 ):
        """
        TransContact class constructor for
        generating reproducible labels.
        :param chromA: str, first chromosome for the trans-chromosomal pair.
        :param chromB: str, second chromosome for the trans-chromosomal pair.
        :param chromA_len: int, length of the first chromosome.
        :param chromB_len: int, length of the second chromosome.
        :param resolution: int, resolution of the Hi-C data.
        :param rep_name: str, name of the experiment (tissue/cell line)
        :param hic_files: list, list of paths to the hic/cool files.
        :param reads_threshold_pos: float, read threshold for positive labels.
        :param reads_threshold_neg: float, read threshold for negative labels.
        """
        # Names of the interacting
        # chromosome pair
        self.chromA = chromA
        self.chromB = chromB

        # lengths of interacting
        # chromosome pair
        self.chromA_len = chromA_len
        self.chromB_len = chromB_len

        # current resolution
        self.resolution = resolution

        # reproducibility within
        # tissue or between tissues
        # along with name of tissue
        self.rep_name = rep_name
        self.name = f"{self.chromA}_{self.chromB}_{self.rep_name}"

        # list of all hic files
        # to be used in reproducible
        # contact prediction
        self.hic_files = hic_files

        self.reads_threshold_pos = reads_threshold_pos

        # upper threshold of reads we need for negative label
        self.reads_threshold_neg = reads_threshold_neg

        # Indicator matrix of Hi-C contacts
        # that meet reproducible threshold
        # for positive label
        self.hic_pos_num = None

        # Indicator matrix of Hi-C contacts
        # that meet the reproducible threshold
        # for negative label
        self.hic_neg_num = None

        # sum of all Hi-C contacts reads
        self.hic_sum = None

        self.artifact_ranking = dict()

        self.hic_sum_noartifacts = None

        self.top_row_artifact_threshold = None
        self.bottom_row_artifact_threshold = None

        self.top_row_artifacts = None
        self.bottom_row_artifact = None

        self.top_col_artifact_threshold = None
        self.bottom_col_artifact_threshold = None

        self.top_col_artifacts = None
        self.bottom_col_artifact = None

        self.pos_samples = None

        self.neg_samples = None

        self.top_percentile = None
        self.bottom_percentile = None

        self.nansum_thr_pos = None
        self.total_notnan_frac_pos = None

        self.nansum_thr_neg = None
        self.total_notnan_frac_neg = None

        self.cnt_pos = None
        self.cnt_neg = None

    def extract_reproducible_contacts(self,
                                      hic_datatype="observed",
                                      hic_raw="NONE",
                                      hic_norm="KR",
                                      hic_type="BP"):
        """
        Read hic/cool files to extract reproducible labels.
        :param hic_datatype: str, observed or observed/expected.
        :param hic_raw: str, use NONE if raw data is desired.
        :param hic_norm: str, Hi-C normalization, e.g., KR
        :param hic_type: str, Basepair or bins
        :return: None
        """
        # For all Hi-C files
        for k in range(len(self.hic_files)):
            # Get chromosome coordinates
            startA = 0
            endA = int(self.chromA_len)

            startB = 0
            endB = int(self.chromB_len)

            print(f"{self.chromA}, startA, endA: {startA, endA}")
            print(f"{self.chromB}, startB, endB: {startB, endB}")

            # Read the cooler file
            hic = cooler.Cooler(f"{self.hic_files[k]}::resolutions//{self.resolution}")

            # Extract raw Hi-C object
            hic_obj_raw = hic.matrix(balance=False)

            # extract trans-contact array for a chromosome pair from raw Hi-C object
            hic_values_raw = hic_obj_raw.fetch(f"{self.chromA}:{startA}-{endA}",
                                               f"{self.chromB}:{startB}-{endB}")

            print(f"hic_values_raw: {hic_values_raw.shape[0] * hic_values_raw.shape[1]}")

            print(f"NaN hic_values_raw: {np.sum(np.isnan(hic_values_raw))}")

            # compute at what quantile is the read threshold for positives
            raw_quantile_pos = (hic_values_raw <= self.reads_threshold_pos).mean()
            print(f"Reads threshold for positive label: {self.reads_threshold_pos}")
            print(f"Reads quantile for positive label: {raw_quantile_pos}")

            # compute at what quantile is the read theshold for negatives
            raw_quantile_neg = (hic_values_raw <= self.reads_threshold_neg).mean()
            print(f"Reads threshold for negative label: {self.reads_threshold_neg}")
            print(f"Reads quantile for negative label: {raw_quantile_neg}")

            hic_values = normalize_hic_data(hic,
                                            fetch_str=f"{self.chromA}:{startA}-{endA}",
                                            region_2=f"{self.chromB}:{startB}-{endB}",
                                            resolution=self.resolution,
                                            kernel_stddev=0.0)

            # hic_obj = hic.matrix(balance=True)

            # hic_values = hic_obj.fetch(f"{self.chromA}:{startA}-{endA}",
            #                           f"{self.chromB}:{startB}-{endB}")

            print(f"hic_values: {hic_values.shape[0] * hic_values.shape[1]}")
            print(f"NaN hic_values: {np.sum(np.isnan(hic_values))}")

            hic_values_quant = np.array(hic_values, copy=True)
            # hic_values_quant[np.isnan(hic_values_quant)] = 0.0
            print(f"hic_values_quant: {hic_values_quant.shape}")

            hic_values_quant = hic_values_quant[~np.isnan(hic_values_quant)]
            print(f"hic_values_quant: {hic_values_quant.shape}")

            # extract the normalized reads corresponding the raw reads
            norm_quantile_pos = np.quantile(hic_values_quant, raw_quantile_pos)
            print(f"Norm Reads quantile for positive label: {norm_quantile_pos}")

            # extract the normalized reads corresponding the raw reads
            norm_quantile_neg = np.quantile(hic_values_quant, raw_quantile_neg)
            print(f"Norm quantile for negative label: {norm_quantile_neg}")

            # If first sample
            if k == 0:
                # find which contacts have more than norm_quantile reads
                self.hic_pos_num = np.where(hic_values >= norm_quantile_pos, 1.0, 0.0)
                self.hic_neg_num = np.where(hic_values <= norm_quantile_neg, 1.0, 0.0)
                # and retain their sum
                self.hic_sum = hic_values
            else:
                # find which contacts have more than norm_quantile reads
                self.hic_pos_num += np.where(hic_values >= norm_quantile_pos, 1.0, 0.0)
                self.hic_neg_num += np.where(hic_values <= norm_quantile_neg, 1.0, 0.0)
                # maintain sum of all reads
                self.hic_sum += hic_values
            self.hic_pos_num[np.isnan(self.hic_sum)] = np.nan
            self.hic_neg_num[np.isnan(self.hic_sum)] = np.nan

            print(f"self.hic_pos_num: {np.nansum(self.hic_pos_num)}, {np.sum(np.isnan(self.hic_pos_num))}")
            print(f"self.hic_neg_num: {np.nansum(self.hic_neg_num)}, {np.sum(np.isnan(self.hic_neg_num))}")

    def remove_top_bottom_artifacts(self,
                                    top_percentile,
                                    bottom_percentile):
        """
        Remove artifact rows and columns based on total count of reads.
        :param top_percentile: float, top percentile of rows/columns to remove.
        :param bottom_percentile: float, bottom percentile of rows/columns to remove.
        :return: None.
        """
        print("In remove_top_bottom_artifacts")
        self.top_percentile = top_percentile
        self.bottom_percentile = bottom_percentile

        # Extract row and column artifact ranking
        row_artifacts = self.artifact_ranking[self.chromA]
        col_artifacts = self.artifact_ranking[self.chromB]

        # Get the row artifact threshold
        self.top_row_artifact_threshold = np.quantile(row_artifacts, self.top_percentile)
        self.bottom_row_artifact_threshold = np.quantile(row_artifacts, self.bottom_percentile)

        # Get the row artifacts
        self.top_row_artifacts = np.where(row_artifacts >= self.top_row_artifact_threshold)[0]
        self.bottom_row_artifact = np.where(row_artifacts <= self.bottom_row_artifact_threshold)[0]

        # Get the column artifact threshold
        self.top_col_artifact_threshold = np.quantile(col_artifacts, self.top_percentile)
        self.bottom_col_artifact_threshold = np.quantile(col_artifacts, self.bottom_percentile)

        # Get the column artifact threshold
        self.top_col_artifacts = np.where(col_artifacts >= self.top_col_artifact_threshold)[0]
        self.bottom_col_artifact = np.where(col_artifacts <= self.bottom_col_artifact_threshold)[0]

        # get the array where we will keep sum of counts but
        # make top and bottom row/column artifacts as NaNs.
        self.hic_sum_noartifacts = np.array(self.hic_sum, copy=True)

        self.hic_sum_noartifacts[self.top_row_artifacts, :] = np.nan
        self.hic_sum_noartifacts[self.bottom_row_artifact, :] = np.nan

        self.hic_sum_noartifacts[:, self.top_col_artifacts] = np.nan
        self.hic_sum_noartifacts[:, self.bottom_col_artifact] = np.nan

    def remove_pos_artifacts(self,
                             nansum_thr_pos=0.9,
                             total_notnan_frac_pos=0.9):
        """
        Remove positive artifacts that have too many contacts at the
        chromosome level.
        :param nansum_thr_pos: float, fraction of contacts.
        :param total_notnan_frac_pos: float, fraction of not nan contacts
        :return: None
        """
        print("In remove_pos_artifacts")
        self.nansum_thr_pos = nansum_thr_pos
        self.total_notnan_frac_pos = total_notnan_frac_pos

        # find artifact rows
        for l in range(self.hic_pos_num_sup.shape[0]):
            num_counts = len(np.where(self.hic_pos_num_sup[l, :] == 1)[0])
            if np.nansum(self.hic_pos_num_sup[l, :]) > self.nansum_thr_pos * len(self.hic_pos_num_sup[l, :]):
                frac_counts = num_counts / float(np.nansum(self.hic_pos_num_sup[l, :]))
                if frac_counts > self.total_notnan_frac_pos:
                    self.hic_pos_num_sup[l, :] = np.nan

        # find artifact columns
        for l in range(self.hic_pos_num_sup.shape[1]):
            num_counts = len(np.where(self.hic_pos_num_sup[:, l] == 1)[0])
            if np.nansum(self.hic_pos_num_sup[:, l]) > self.nansum_thr_pos * len(self.hic_pos_num_sup[:, l]):
                frac_counts = num_counts / float(np.nansum(self.hic_pos_num_sup[:, l]))

                if frac_counts > self.total_notnan_frac_pos:
                    self.hic_pos_num_sup[:, l] = np.nan

    def remove_neg_artifacts(self,
                             nansum_thr_neg=0.4,
                             total_notnan_frac_neg=0.9):
        """
        Remove negative artifacts that have no contacts or less than
        the specified fraction of no contacts. Empty loci are probably
        repeat regions.
        :param nansum_thr_neg: float, fraction of no contacts.
        :param total_notnan_frac_neg: float, fraction of not nans.
        :return: None
        """
        print("In remove_neg_artifacts")
        self.nansum_thr_neg = nansum_thr_neg
        self.total_notnan_frac_neg = total_notnan_frac_neg

        # find artifact rows
        for l in range(self.hic_neg_num_sup.shape[0]):
            num_counts = len(np.where(self.hic_neg_num_sup[l, :] == 1)[0])
            if np.nansum(self.hic_neg_num_sup[l, :]) > self.nansum_thr_neg * len(self.hic_neg_num_sup[l, :]):
                frac_counts = num_counts / float(np.nansum(self.hic_neg_num_sup[l, :]))

                if frac_counts > self.total_notnan_frac_neg:
                    self.hic_neg_num_sup[l, :] = np.nan

        # find artifact columns
        for l in range(self.hic_neg_num_sup.shape[1]):
            num_counts = len(np.where(self.hic_neg_num_sup[:, l] == 1)[0])
            if np.nansum(self.hic_neg_num_sup[:, l]) > self.nansum_thr_neg * len(self.hic_neg_num_sup[:, l]):
                frac_counts = num_counts / float(np.nansum(self.hic_neg_num_sup[:, l]))

                if frac_counts > self.total_notnan_frac_neg:
                    self.hic_neg_num_sup[:, l] = np.nan

    def make_supervised_labels(self,
                               pos_samples,
                               neg_samples,
                               top_percentile,
                               bottom_percentile,
                               nansum_thr_pos=0.9,
                               total_notnan_frac_pos=0.9,
                               nansum_thr_neg=0.4,
                               total_notnan_frac_neg=0.9):
        """
        Make robust supervised labels after removing artifacts.
        :param pos_samples: int, number of positive samples.
        :param neg_samples: int, number of negative samples.
        :param top_percentile: float, top percentile by count to remove genomic locus genome-wide.
        :param bottom_percentile: float, bottom percentile by count to remove to remove genomic locus genome-wide.
        :param nansum_thr_pos: float, positive threshold for contacts to remove at trans-chromosome level.
        :param total_notnan_frac_pos: float, not nan fraction of contacts for a genomic locus with another chromosome.
        :param nansum_thr_neg: float, negative threshold for no contact to remove at trans-chromosome level/
        :param total_notnan_frac_neg: float, not nan fraction of no contacts for a genomic locus with another chromosome.
        :return: None
        """
        print("In make_supervised_labels")
        # Remove genomic locus which fall in top or bottom percentile by total reads
        self.remove_top_bottom_artifacts(top_percentile, bottom_percentile)
        print(f"self.hic_pos_num: {np.nansum(self.hic_pos_num)}")
        print(f"self.hic_neg_num: {np.nansum(self.hic_neg_num)}")

        # Compute robust positive loci
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.hic_pos_num_sup = np.where(self.hic_pos_num >= self.pos_samples, 1.0, 0.0)
        print(f"self.hic_pos_num_sup: {np.nansum(self.hic_pos_num_sup)}")

        # Make loci that don't pass threshold as NaN.
        self.hic_pos_num_sup[np.isnan(self.hic_pos_num)] = np.nan

        # Make top/bottom row/column artifacts by total reads as NaNs.
        self.hic_pos_num_sup[self.top_row_artifacts, :] = np.nan
        self.hic_pos_num_sup[self.bottom_row_artifact, :] = np.nan

        self.hic_pos_num_sup[:, self.top_col_artifacts] = np.nan
        self.hic_pos_num_sup[:, self.bottom_col_artifact] = np.nan

        # Compute robust negative loci
        self.hic_neg_num_sup = np.where(self.hic_neg_num >= self.neg_samples, 1.0, 0.0)
        print(f"self.hic_neg_num_sup: {np.nansum(self.hic_neg_num_sup)}")

        # Make loci that don't pass threshold as NaN.
        self.hic_neg_num_sup[np.isnan(self.hic_neg_num)] = np.nan

        # Make top/bottom row/column artifacts by total reads as NaNs.
        self.hic_neg_num_sup[self.top_row_artifacts, :] = np.nan
        self.hic_neg_num_sup[self.bottom_row_artifact, :] = np.nan

        self.hic_neg_num_sup[:, self.top_col_artifacts] = np.nan
        self.hic_neg_num_sup[:, self.bottom_col_artifact] = np.nan

        # Remove artifacts at chromosome level
        self.remove_pos_artifacts(nansum_thr_pos, total_notnan_frac_pos)
        print(f"self.hic_pos_num_sup: {np.nansum(self.hic_pos_num_sup)}")
        print(f"self.hic_neg_num_sup: {np.nansum(self.hic_neg_num_sup)}")

        # Remove artifacts at chromosome level
        self.remove_neg_artifacts(nansum_thr_neg, total_notnan_frac_neg)
        print(f"self.hic_pos_num_sup: {np.nansum(self.hic_pos_num_sup)}")
        print(f"self.hic_neg_num_sup: {np.nansum(self.hic_neg_num_sup)}")

        self.cnt_pos = np.nansum(self.hic_pos_num_sup)
        self.cnt_neg = np.nansum(self.hic_neg_num_sup)

    def save_obj(self, save_path):
        """
        Save TransContact as a pickle file.
        :param save_path: str, path to saved file
        :return: None.
        """
        with gzip.open(save_path, 'wb') as save_obj:
            pickle.dump(self, save_obj)


class GenomeWideTransContacts():
    """
    Keep track of trans chromosomal pairs and
    generate labels after artifact removal.
    """

    def __init__(self,
                 config_file):
        """
        Constructor for GenomeWideTransContact object.
        :param config_file: str, path to config file
        """
        # make a config parser object
        self.config = configparser.ConfigParser()

        # read its parameters
        self.config.read(config_file)

        # get the condition (within tissue/cross tissue)
        # in which we are computing reproducible contacts
        self.rep_name = self.config['data_parameters']['rep_name']

        # get list of hi-c replicates
        self.hic_files_repgrp = np.array(
            decode_list(self.config['input_files']['hic_files_repgrp']),
            dtype=str
        )

        # Resolution we are working at
        self.selected_resolution = int(self.config['data_parameters']['selected_resolution'])

        # lower theshold of reads we need for positive label
        self.reads_threshold_pos = float(self.config['data_parameters']['reads_threshold_pos'])

        # upper threshold of reads we need for negative label
        self.reads_threshold_neg = float(self.config['data_parameters']['reads_threshold_neg'])

        # chromosome order
        self.chroms_order = decode_chrome_order_inv_dict(
            self.config['data_parameters']['chroms_order_inv'])

        self.save_prefix = self.config['output_files']['save_prefix']

        self.labels_file = self.config['output_files']['labels_file']

        # Chromosome names
        self.chrom_names = []

        # Chromosome lengths
        self.chrom_lengths = {}

        self.artifact_ranking = dict()

        self.pos_samples = float(self.config['data_parameters']['pos_samples'])
        self.neg_samples = float(self.config['data_parameters']['neg_samples'])
        self.top_percentile = float(self.config['data_parameters']['top_percentile'])
        self.bottom_percentile = float(self.config['data_parameters']['bottom_percentile'])
        self.nansum_thr_pos = float(self.config['data_parameters']['nansum_thr_pos'])
        self.total_notnan_frac_pos = float(self.config['data_parameters']['total_notnan_frac_pos'])
        self.nansum_thr_neg = float(self.config['data_parameters']['nansum_thr_neg'])
        self.total_notnan_frac_neg = float(self.config['data_parameters']['total_notnan_frac_neg'])
        self.hic_norm = self.config['data_parameters']['hic_norm']
        self.hic_raw = self.config['data_parameters']['hic_raw']
        self.hic_datatype = self.config['data_parameters']['hic_datatype']
        self.hic_type = self.config['data_parameters']['hic_type']

        # get list of hi-c replicates
        self.train_chroms = np.array(
            decode_list(self.config['data_parameters']['train_chroms']),
            dtype=str
        )

        # get list of hi-c replicates
        self.val_chroms = np.array(
            decode_list(self.config['data_parameters']['val_chroms']),
            dtype=str
        )

        # get list of hi-c replicates
        self.test_chroms = np.array(
            decode_list(self.config['data_parameters']['test_chroms']),
            dtype=str
        )

    def extract_gw_trans_contacts(self):

        """
        Extract chromosome names and
        lengths from an HiC file.
        :return: None.
        """
        # Get Chromosome names and lengths from the cooler file
        self.chrom_lengths = cooler.util.fetch_chromsizes('hg38').to_dict()
        # We don't use chrY and mitochondrial DNA
        del self.chrom_lengths['chrM']
        del self.chrom_lengths['chrY']
        self.chrom_names = list(self.chrom_lengths.keys())

        # For all trans-chromosome pairs
        for i in range(1, len(self.chroms_order), 1):
            for j in range(i + 1, len(self.chroms_order) + 1, 1):
                chromA = self.chroms_order[i]
                chromB = self.chroms_order[j]
                print(chromA, chromB)
                save_path = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}.gz"
                # Make TransContact object for this trans-chromosome pair if it doesn't exist.
                if not os.path.exists(save_path):
                    trans_contacts = TransContacts(
                        chromA,
                        chromB,
                        chromA_len=self.chrom_lengths[chromA],
                        chromB_len=self.chrom_lengths[chromB],
                        resolution=self.selected_resolution,
                        rep_name=self.rep_name,
                        hic_files=self.hic_files_repgrp,
                        reads_threshold_pos=self.reads_threshold_pos,
                        reads_threshold_neg=self.reads_threshold_neg
                    )
                    # Extract reproducible contacts for this trans-chromosome pair.
                    trans_contacts.extract_reproducible_contacts(hic_datatype=self.hic_datatype,
                                                                 hic_norm=self.hic_norm,
                                                                 hic_raw=self.hic_raw,
                                                                 hic_type=self.hic_type)
                    trans_contacts.save_obj(save_path)
                else:
                    print(f"{save_path} exists, delete it to rerun.")

    def compute_gw_artifacts(self):
        """
        Compute genome-wide artifacts and remove them.
        :return: None.
        """
        # For trans-chromosome pair
        for i in range(1, len(self.chroms_order) + 1, 1):
            chromA = self.chroms_order[i]
            flipped = False
            base_chom_rows = None
            for j in range(1, len(self.chroms_order) + 1, 1):
                chromB = self.chroms_order[j]
                if chromA != chromB:
                    save_path = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}.gz"
                    # For chromosome pairs in the lower triangle of the genome-wide contact map, make the Trans
                    # contact object and make sure to set flipped to true.
                    if not os.path.exists(save_path):
                        save_path = f"{self.save_prefix}/{chromB}_{chromA}_{self.rep_name}_{self.selected_resolution}.gz"
                        flipped = True
                    # Open the TransContact object
                    with gzip.open(save_path, 'rb') as save_obj:
                        print(save_path)
                        trans_contacts = pickle.load(save_obj)
                        # If it needs to be flipped
                        if flipped:
                            flipped = False
                            # If this is the first chromosome pair for this chromosome A.
                            if base_chom_rows is None:
                                # flip it.
                                base_chom_rows = trans_contacts.hic_sum.T
                                print(f"flipped: base_chom_rows: {base_chom_rows.shape}")
                            # If this is the first chromosome pair for this chromosome A.
                            else:
                                print(f"flipped: trans_contacts.hic_sum.T: {trans_contacts.hic_sum.T.shape}")
                                base_chom_rows = np.concatenate((base_chom_rows,
                                                                 trans_contacts.hic_sum.T), axis=1)
                            print(f"flipped: base_chom_rows: {base_chom_rows.shape}")
                        # If no flipping needed
                        else:
                            if base_chom_rows is None:
                                base_chom_rows = trans_contacts.hic_sum
                                print(f"Not flipped: base_chom_rows: {base_chom_rows.shape}")
                            else:
                                print(f"Not flipped trans_contacts.hic_sum: {trans_contacts.hic_sum.shape}")
                                base_chom_rows = np.concatenate((base_chom_rows,
                                                                 trans_contacts.hic_sum), axis=1)
                            print(f"Not flipped: base_chom_rows: {base_chom_rows.shape}")

            base_chom_sum = np.sum(base_chom_rows, axis=1)
            print(f"base_chom_sum: {base_chom_sum.shape}")

            self.artifact_ranking[chromA] = base_chom_sum

        # For all trans-chromosome pairs
        for i in range(1, len(self.chroms_order), 1):
            for j in range(i + 1, len(self.chroms_order) + 1, 1):
                chromA = self.chroms_order[i]
                chromB = self.chroms_order[j]
                # get the TransContact object
                save_path = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}.gz"
                # Open it.
                with gzip.open(save_path, 'rb') as save_obj:
                    print(chromA, chromB)

                    trans_contacts = pickle.load(save_obj)
                    # Get artifact ranking by total reads for each genomic locus
                    trans_contacts.artifact_ranking[chromA] = self.artifact_ranking[chromA]
                    trans_contacts.artifact_ranking[chromB] = self.artifact_ranking[chromB]

                    trans_contacts.save_obj(save_path)

    def make_gw_supervised_labels(self):
        """
        Make robust supervised labels for all trans-chromosome pairs.
        :return: None.
        """
        cnt_pos = 0
        cnt_neg = 0
        all_chr_contact_bed = open(f"{self.labels_file}_all_{self.rep_name}_{self.selected_resolution}_sup_5x128KB.txt",
                                   'w')
        # For all trans-chromosome pairs
        for i in range(1, len(self.chroms_order), 1):
            for j in range(i + 1, len(self.chroms_order) + 1, 1):
                chromA = self.chroms_order[i]
                chromB = self.chroms_order[j]
                print(chromA, chromB)

                save_path_new = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}.gz"

                # Open the corresponding TransContact object
                with gzip.open(save_path_new, 'rb') as save_obj:
                    print(save_path_new)
                    trans_contacts = pickle.load(save_obj)
                    trans_contacts.make_supervised_labels(
                        self.pos_samples,
                        self.neg_samples,
                        self.top_percentile,
                        self.bottom_percentile,
                        self.nansum_thr_pos,
                        self.total_notnan_frac_pos,
                        self.nansum_thr_neg,
                        self.total_notnan_frac_neg)

                    print(f"hic_pos_num: {np.nansum(trans_contacts.hic_pos_num_sup)}")
                    print(f"hic_neg_num: {np.nansum(trans_contacts.hic_neg_num_sup)}")
                    chr_contact_bed = open(
                        f"{self.labels_file}_{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_sup_5x128KB.txt",
                        'w')
                    # For each genomic locus pair
                    for xx in range(trans_contacts.hic_pos_num_sup.shape[0]):
                        str_A = f"{trans_contacts.chromA}\t{(xx - 2) * self.selected_resolution}\t{(xx + 3) * self.selected_resolution}"
                        for yy in range(trans_contacts.hic_pos_num_sup.shape[1]):
                            # If label is positive and it is not an artifact
                            if trans_contacts.hic_pos_num_sup[xx, yy] == 1 and not np.isnan(
                                    trans_contacts.hic_sum_noartifacts[xx, yy]):
                                str_B = f"{trans_contacts.chromB}\t{(yy - 2) * self.selected_resolution}\t{(yy + 3) * self.selected_resolution}"
                                tmp = ",".join(
                                    np.array(trans_contacts.hic_sum_noartifacts[xx - 2:xx + 3, yy - 2:yy + 3].flatten(),
                                             dtype=str))
                                write_str = f"{str_A}\t{str_B}\t1\t{tmp}\n"
                                chr_contact_bed.write(write_str)
                                all_chr_contact_bed.write(write_str)
                            # If label is negative and it is not an artifact
                            elif trans_contacts.hic_neg_num_sup[xx, yy] == 1 and not np.isnan(
                                    trans_contacts.hic_sum_noartifacts[xx, yy]):
                                str_B = f"{trans_contacts.chromB}\t{(yy - 2) * self.selected_resolution}\t{(yy + 3) * self.selected_resolution}"
                                tmp = ",".join(
                                    np.array(trans_contacts.hic_sum_noartifacts[xx - 2:xx + 3, yy - 2:yy + 3].flatten(),
                                             dtype=str))
                                write_str = f"{str_A}\t{str_B}\t0\t{tmp}\n"
                                chr_contact_bed.write(write_str)
                                all_chr_contact_bed.write(write_str)
                    # save chromosome level label file
                    chr_contact_bed.close()
                    cnt_pos += np.nansum(trans_contacts.hic_pos_num_sup)
                    cnt_neg += np.nansum(trans_contacts.hic_neg_num_sup)

                    trans_contacts.save_obj(save_path_new)
        # Keep genomewide label file
        all_chr_contact_bed.close()
        print(f"cnt_pos: {cnt_pos}")
        print(f"cnt_neg: {cnt_neg}")

    def compute_replicate_metrics(self,
                                  chrom_set,
                                  chrom_set_name="train_chroms"):
        """
        Compute labels for the replicate data for computing
        reproducibility upper limit.
        :param chrom_set: list, list of chromosomes
        :param chrom_set_name: str, name of chromosome set.
        :return:
        """

        # get list of hi-c replicates
        hic_files_testgrp = np.array(
            decode_list(self.config['input_files']['hic_files_testgrp']),
            dtype=str
        )

        self.chrom_lengths = cooler.util.fetch_chromsizes('hg38').to_dict()
        del self.chrom_lengths['chrM']
        del self.chrom_lengths['chrY']
        self.chrom_names = list(self.chrom_lengths.keys())

        cnt_pos = 0
        cnt_neg = 0
        all_labels = []
        all_preds = []
        for i in range(1, len(self.chroms_order), 1):
            for j in range(i + 1, len(self.chroms_order) + 1, 1):
                chromA = self.chroms_order[i]
                chromB = self.chroms_order[j]
                print(chromA, chromB)
                if 'chr' not in chromA:
                    chromA_set = f"chr{chromA}"
                    chromB_set = f"chr{chromB}"
                else:
                    chromA_set = chromA
                    chromB_set = chromB
                if chromA_set in chrom_set and chromB_set in chrom_set:
                    save_path_sup = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}.gz"
                    with gzip.open(save_path_sup, 'rb') as save_obj:
                        trans_contacts = pickle.load(save_obj)
                        trans_contacts.hic_pos_num_sup
                        trans_contacts.hic_neg_num_sup

                        startA = 0
                        endA = int(self.chrom_lengths[chromA])

                        startB = 0
                        endB = int(self.chrom_lengths[chromB])

                        print(f"startA, endA: {startA, endA}")
                        print(f"startB, endB: {startB, endB}")

                        hic = cooler.Cooler(f"{hic_files_testgrp[0]}::resolutions//{self.selected_resolution}")

                        hic_obj = hic.matrix(balance=True)

                        hic_values = hic_obj.fetch(f"{chromA}:{startA}-{endA}",
                                                   f"{chromB}:{startB}-{endB}")

                        pos_idx = np.where(trans_contacts.hic_pos_num_sup.flatten() == 1.0)[0]
                        neg_idx = np.where(trans_contacts.hic_neg_num_sup.flatten() == 1.0)[0]
                        print(f"positives: {len(pos_idx)}")
                        print(f"negatives: {len(neg_idx)}")

                        labels = np.zeros((len(pos_idx) + len(neg_idx)))
                        labels[0:len(pos_idx)] = 1.0

                        preds = np.concatenate((hic_values.flatten()[pos_idx],
                                                hic_values.flatten()[neg_idx]))

                        all_labels.extend(labels)
                        all_preds.extend(preds)

        np.savez(f"{self.save_prefix}/{chrom_set_name}_{self.rep_name}_"
                 f"{self.selected_resolution}_replicate.npz",
                 all_labels=all_labels,
                 all_preds=all_preds)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file", type=str, default="config_data_within_tissue.yml", help="path to the config file."
    )

    args = parser.parse_args()

    gw_trans_contact = GenomeWideTransContacts(args.config_file)

    gw_trans_contact.extract_gw_trans_contacts()
    gw_trans_contact.compute_gw_artifacts()
    gw_trans_contact.make_gw_supervised_labels()
    gw_trans_contact.compute_replicate_metrics(gw_trans_contact.val_chroms,
                                               chrom_set_name="val_chroms")
    gw_trans_contact.compute_replicate_metrics(gw_trans_contact.test_chroms,
                                               chrom_set_name="test_chroms")


if __name__ == "__main__":
    main()
