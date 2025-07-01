"""
twinc_data.py
Author: Anupama Jha <anupamaj@uw.edu>
In this script, we generate labels for
trans-contacts
"""

import gzip
import hicstraw
import numpy as np
import _pickle as pickle


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
        TransContact class for every chromosome.
        This class keeps track of reads from all
        the replicates.
        :param chromA: str, chromosome A
        :param chromB: str, chromosome B
        :param chromA_len: int, length of chromosome A
        :param chromB_len: int, length of chromosome B
        :param resolution: int, Hi-C data resolution
                                default=100KB
        :param rep_name: str, name of the experiment,
                              usually the tissue/cell line.
        :param hic_files: list, list of replicate hi-c files.
        :param reads_threshold_pos: float, positive read threshold.
        :param reads_threshold_neg: float, negative read threshold.
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
        Extract genomic locus pairs, which have more reads
        than the positive threshold for at least the number
        of required donors as positive labels and the locus
        pairs, which have fewer reads than the negative
        threshold for at least the number of required donors
        as negative labels.
        :param hic_datatype: str, observed or observed/expected
        :param hic_raw: str, if raw Hi-C needed use NONE
        :param hic_norm: str, if norm needed, e.g., KR
        :param hic_type: str, type of Hi-C entry, e.g., BP
        :return: None
        """
        for k in range(len(self.hic_files)):
            startA = 0
            endA = int(self.chromA_len)

            startB = 0
            endB = int(self.chromB_len)

            print(f"startA, endA: {startA, endA}")
            print(f"startB, endB: {startB, endB}")

            print(f"hic: {self.hic_files[k]}, {self.chromA}, {self.chromB}")

            # Get Hi-C object
            hic = hicstraw.HiCFile(self.hic_files[k])

            chrA_name = self.chromA  # .replace("chr", "")
            chrB_name = self.chromB  # .replace("chr", "")

            # Get raw Hi-C reads
            hic_obj_raw = hic.getMatrixZoomData(chrA_name,
                                                chrB_name,
                                                hic_datatype,
                                                hic_raw,
                                                hic_type,
                                                self.resolution)
            # extract a patch
            hic_values_raw = hic_obj_raw.getRecordsAsMatrix(startA,
                                                            endA,
                                                            startB,
                                                            endB)

            print(f"hic_values_raw: {hic_values_raw}")

            print(f"NaN hic_values_raw: {np.sum(np.isnan(hic_values_raw))}")

            # compute at what quantile is the read threshold for positives
            raw_quantile_pos = (hic_values_raw <= self.reads_threshold_pos).mean()
            print(f"Reads threshold for positive label: {self.reads_threshold_pos}")
            print(f"Reads quantile for positive label: {raw_quantile_pos}")

            # compute at what quantile is the read theshold for negatives
            raw_quantile_neg = (hic_values_raw <= self.reads_threshold_neg).mean()
            print(f"Reads threshold for negative label: {self.reads_threshold_neg}")
            print(f"Reads quantile for negative label: {raw_quantile_neg}")

            chrA_name = self.chromA  # .replace("chr", "")
            chrB_name = self.chromB  # .replace("chr", "")

            # Get scale normalized data
            hic_obj = hic.getMatrixZoomData(chrA_name,
                                            chrB_name,
                                            hic_datatype,
                                            hic_norm,
                                            hic_type,
                                            self.resolution)
            # extract a patch
            hic_values = hic_obj.getRecordsAsMatrix(startA,
                                                    endA,
                                                    startB,
                                                    endB)

            print(f"hic_values: {np.sum(np.isnan(hic_values))}")
            print(f"NaN hic_values: {np.sum(np.isnan(hic_values))}")

            # extract the normalized reads corresponding the raw reads
            norm_quantile_pos = np.quantile(hic_values, raw_quantile_pos)
            print(f"Norm Reads quantile for positive label: {norm_quantile_pos}")

            # extract the normalized reads corresponding the raw reads
            norm_quantile_neg = np.quantile(hic_values, raw_quantile_neg)
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
            print(f"self.hic_pos_num: {np.nansum(self.hic_pos_num)}")
            print(f"self.hic_neg_num: {np.nansum(self.hic_neg_num)}")

    def remove_top_bottom_artifacts(self,
                                    top_percentile,
                                    bottom_percentile):
        """
        Remove the top and bottom artifacts by read count.
        :param top_percentile: float, top percentile for removal
        :param bottom_percentile: flaot, bottom percentile for removal
        :return: None
        """
        print("In remove_top_bottom_artifacts")
        self.top_percentile = top_percentile
        self.bottom_percentile = bottom_percentile

        row_artifacts = self.artifact_ranking[self.chromA]
        col_artifacts = self.artifact_ranking[self.chromB]

        self.top_row_artifact_threshold = np.quantile(row_artifacts, self.top_percentile)
        self.bottom_row_artifact_threshold = np.quantile(row_artifacts, self.bottom_percentile)

        self.top_row_artifacts = np.where(row_artifacts >= self.top_row_artifact_threshold)[0]
        self.bottom_row_artifact = np.where(row_artifacts <= self.bottom_row_artifact_threshold)[0]

        self.top_col_artifact_threshold = np.quantile(col_artifacts, self.top_percentile)
        self.bottom_col_artifact_threshold = np.quantile(col_artifacts, self.bottom_percentile)

        self.top_col_artifacts = np.where(col_artifacts >= self.top_col_artifact_threshold)[0]
        self.bottom_col_artifact = np.where(col_artifacts <= self.bottom_col_artifact_threshold)[0]

        self.hic_sum_noartifacts = np.array(self.hic_sum, copy=True)

        self.hic_sum_noartifacts[self.top_row_artifacts, :] = np.nan
        self.hic_sum_noartifacts[self.bottom_row_artifact, :] = np.nan

        self.hic_sum_noartifacts[:, self.top_col_artifacts] = np.nan
        self.hic_sum_noartifacts[:, self.bottom_col_artifact] = np.nan

    def remove_pos_artifacts(self,
                             nansum_thr_pos=0.9,
                             total_notnan_frac_pos=0.9):
        """
        Remove genomic locus with contacts across the entire
        chromosome if the number of contacts is greater than
        the specified threshold, as these are likely to be
        artifacts from Hi-C experiment.
        :param nansum_thr_pos: float, fraction of contacts
        :param total_notnan_frac_pos: float, fraction of
                                      not nan values for
                                      a row/column.
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
        Remove genomic loci that don't interact with other
        regions as these are likely to be repeat region and
        we cannot map to them.
        :param nansum_thr_neg: float, fraction of no contacts
        :param total_notnan_frac_neg: float, fraction of not
                                             nan values.
        :return: None.
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
        Given the object with top, bottom and chromosome level
        artifacts filtered out, we will now write the supervised
        labels.
        :param pos_samples: int, number of positive samples.
        :param neg_samples: int, number of negative samples.
        :param top_percentile: float, top percentile for removal.
        :param bottom_percentile: flaot, bottom percentile for removal.
        :param nansum_thr_pos: float, fraction of contacts for positive
                                      row/column removal at chromosome level.
        :param total_notnan_frac_pos: float, fraction of not nans for
                                     positive row/column removal at chromosome level.
        :param nansum_thr_neg: float, fraction of contacts for negative
                                      row/column removal at chromosome level.
        :param total_notnan_frac_neg: float, fraction of not nans for
                                     positive row/column removal at chromosome level.
        :return: None
        """
        print("In make_supervised_labels")

        self.remove_top_bottom_artifacts(top_percentile, bottom_percentile)
        print(f"self.hic_pos_num: {np.nansum(self.hic_pos_num)}")
        print(f"self.hic_neg_num: {np.nansum(self.hic_neg_num)}")

        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.hic_pos_num_sup = np.where(self.hic_pos_num >= self.pos_samples, 1.0, 0.0)
        print(f"self.hic_pos_num_sup: {np.nansum(self.hic_pos_num_sup)}")

        self.hic_pos_num_sup[self.top_row_artifacts, :] = np.nan
        self.hic_pos_num_sup[self.bottom_row_artifact, :] = np.nan

        self.hic_pos_num_sup[:, self.top_col_artifacts] = np.nan
        self.hic_pos_num_sup[:, self.bottom_col_artifact] = np.nan

        self.hic_neg_num_sup = np.where(self.hic_neg_num >= self.neg_samples, 1.0, 0.0)
        print(f"self.hic_neg_num_sup: {np.nansum(self.hic_neg_num_sup)}")

        self.hic_neg_num_sup[self.top_row_artifacts, :] = np.nan
        self.hic_neg_num_sup[self.bottom_row_artifact, :] = np.nan

        self.hic_neg_num_sup[:, self.top_col_artifacts] = np.nan
        self.hic_neg_num_sup[:, self.bottom_col_artifact] = np.nan

        self.remove_pos_artifacts(nansum_thr_pos, total_notnan_frac_pos)
        print(f"self.hic_pos_num_sup: {np.nansum(self.hic_pos_num_sup)}")
        print(f"self.hic_neg_num_sup: {np.nansum(self.hic_neg_num_sup)}")

        self.remove_neg_artifacts(nansum_thr_neg, total_notnan_frac_neg)
        print(f"self.hic_pos_num_sup: {np.nansum(self.hic_pos_num_sup)}")
        print(f"self.hic_neg_num_sup: {np.nansum(self.hic_neg_num_sup)}")

        self.cnt_pos = np.nansum(self.hic_pos_num_sup)
        self.cnt_neg = np.nansum(self.hic_neg_num_sup)

    def save_obj(self, save_path):
        """
        Save pickle object, one per chromosome pair.
        :param save_path: str, path for saving.
        :return: None.
        """
        with gzip.open(save_path, 'wb') as save_obj:
            pickle.dump(self, save_obj)


class GenomeWideTransContacts():
    """
    Genome wide trans-contact class for managing
    all chromosome pairs.
    """

    def __init__(self,
                 config_file,
                 replicate_group):
        """
        Constructor
        :param config_file: str, path to config file
        :param replicate_group:  str, whether we are
                                      processing the
                                      replicate group.
        """
        # make a config parser object
        self.config = configparser.ConfigParser()

        # read its parameters
        self.config.read(config_file)

        self.replicate_group = replicate_group

        # get the condition (within tissue/cross tissue)
        # in which we are computing reproducible contacts
        self.rep_name = self.config['data_parameters']['rep_name']

        if self.replicate_group == "No":
            # get list of hi-c replicates
            self.hic_files_repgrp = np.array(
                decode_list(self.config['input_files']['hic_files_repgrp']),
                dtype=str
            )
        else:
            self.hic_files_repgrp = np.array(
                decode_list(self.config['input_files']['hic_files_testgrp']),
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
        """
        # get chromosome names and lengths
        hic = hicstraw.HiCFile(self.hic_files_repgrp[0])
        for chrom in hic.getChromosomes():
            print(chrom.name, chrom.length)
            self.chrom_names.append(chrom.name)
            self.chrom_lengths[chrom.name] = chrom.length

        # for all trans chromosome pairs
        for i in range(1, len(self.chroms_order), 1):
            for j in range(i + 1, len(self.chroms_order) + 1, 1):
                chromA = self.chroms_order[i]
                chromB = self.chroms_order[j]
                print(chromA, chromB)
                # If processing label group
                if self.replicate_group == "No":
                    save_path = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5.gz"
                # If processing repliacte group
                else:
                    save_path = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5_replicate.gz"
                # If TransContact object doesn't exist
                if not os.path.exists(save_path):
                    # Make one
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
                    # get reproducible contacts
                    trans_contacts.extract_reproducible_contacts(hic_datatype=self.hic_datatype,
                                                                 hic_norm=self.hic_norm,
                                                                 hic_raw=self.hic_raw,
                                                                 hic_type=self.hic_type)
                    # save it
                    trans_contacts.save_obj(save_path)
                else:
                    print(f"{save_path} exist, delete it to rerun.")

    def compute_gw_artifacts(self):
        """
        Compute top and bottom artifacts based on total genome-wide contact count for genomic locus.
        :return: None
        """
        # for all chromosomes
        for i in range(1, len(self.chroms_order) + 1, 1):
            chromA = self.chroms_order[i]
            flipped = False
            base_chom_rows = None
            for j in range(1, len(self.chroms_order) + 1, 1):
                chromB = self.chroms_order[j]
                # If trans-chromosome pair
                if chromA != chromB:
                    # If processing label group
                    if self.replicate_group == "No":
                        save_path = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5.gz"
                    # If processing replicate group
                    else:
                        save_path = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5_replicate.gz"
                    # If TransContact object doesn't exist.
                    if not os.path.exists(save_path):
                        if self.replicate_group == "No":
                            save_path = f"{self.save_prefix}/{chromB}_{chromA}_{self.rep_name}_{self.selected_resolution}_5_v_5.gz"
                        else:
                            save_path = f"{self.save_prefix}/{chromB}_{chromA}_{self.rep_name}_{self.selected_resolution}_5_v_5_replicate.gz"
                        # If chromB < chromA in the ordering, we will need to flip it.
                        flipped = True
                    # Open the saved object
                    with gzip.open(save_path, 'rb') as save_obj:
                        print(save_path)
                        trans_contacts = pickle.load(save_obj)
                        # If flipping needed
                        if flipped:
                            flipped = False
                            # If first chrom pair and flipping needed
                            if base_chom_rows is None:
                                # flip
                                base_chom_rows = trans_contacts.hic_sum.T
                                print(f"flipped: base_chom_rows: {base_chom_rows.shape}")
                            # If not first chrom pair, but flipping needed
                            else:
                                print(f"flipped: trans_contacts.hic_sum.T: {trans_contacts.hic_sum.T.shape}")
                                base_chom_rows = np.concatenate((base_chom_rows,
                                                                 trans_contacts.hic_sum.T), axis=1)
                            print(f"flipped: base_chom_rows: {base_chom_rows.shape}")
                        else:
                            # If first chrom pair but no flipping needed
                            if base_chom_rows is None:
                                base_chom_rows = trans_contacts.hic_sum
                                print(f"Not flipped: base_chom_rows: {base_chom_rows.shape}")
                            # If not first chrom pair but no flipping needed
                            else:
                                print(f"Not flipped trans_contacts.hic_sum: {trans_contacts.hic_sum.shape}")
                                base_chom_rows = np.concatenate((base_chom_rows,
                                                                 trans_contacts.hic_sum), axis=1)
                            print(f"Not flipped: base_chom_rows: {base_chom_rows.shape}")

            # get sum along first axis
            base_chom_sum = np.sum(base_chom_rows, axis=1)
            print(f"base_chom_sum: {base_chom_sum.shape}")

            # add sum to the artifact ranking
            self.artifact_ranking[chromA] = base_chom_sum

        # for all chromosome pairs, add artifact ranking
        # to the TransContact object.
        for i in range(1, len(self.chroms_order), 1):
            for j in range(i + 1, len(self.chroms_order) + 1, 1):
                chromA = self.chroms_order[i]
                chromB = self.chroms_order[j]
                if self.replicate_group == "No":
                    save_path = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5.gz"
                else:
                    save_path = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5_replicate.gz"

                with gzip.open(save_path, 'rb') as save_obj:
                    print(chromA, chromB)
                    trans_contacts = pickle.load(save_obj)

                    trans_contacts.artifact_ranking[chromA] = self.artifact_ranking[chromA]
                    trans_contacts.artifact_ranking[chromB] = self.artifact_ranking[chromB]

                    trans_contacts.save_obj(save_path)

    def make_gw_supervised_labels(self):
        """
        Make supervised labels for all trans-chromosome pairs..
        :return: None
        """
        cnt_pos = 0
        cnt_neg = 0
        # If label group
        if self.replicate_group == "No":
            all_chr_contact_bed = open(
                f"{self.labels_file}_all_{self.rep_name}_{self.selected_resolution}_5_v_5_sup.txt", 'w')
        # If replicate group
        else:
            all_chr_contact_bed = open(
                f"{self.labels_file}_all_{self.rep_name}_{self.selected_resolution}_5_v_5_sup_replicate.txt", 'w')
        # For all chromosome pairs
        for i in range(1, len(self.chroms_order), 1):
            for j in range(i + 1, len(self.chroms_order) + 1, 1):
                chromA = self.chroms_order[i]
                chromB = self.chroms_order[j]
                print(chromA, chromB)
                # If label group is being processed
                if self.replicate_group == "No":
                    save_path_new = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5.gz"
                # If replicate group is being processed
                else:
                    save_path_new = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5_replicate.gz"

                # open the trans contact object
                with gzip.open(save_path_new, 'rb') as save_obj:
                    trans_contacts = pickle.load(save_obj)
                    # make supervised labels
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
                    # Make label text file for label group
                    if self.replicate_group == "No":
                        chr_contact_bed = open(
                            f"{self.labels_file}_{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5_sup.txt",
                            'w')
                    # Make label text file for replicate group
                    else:
                        chr_contact_bed = open(
                            f"{self.labels_file}_{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5_sup_replicate.txt",
                            'w')
                    # For every locus pair
                    for xx in range(trans_contacts.hic_pos_num_sup.shape[0]):
                        str_A = f"{trans_contacts.chromA}\t{xx * self.selected_resolution}\t{(xx + 1) * self.selected_resolution}"
                        for yy in range(trans_contacts.hic_pos_num_sup.shape[1]):
                            # add positive label
                            if trans_contacts.hic_pos_num_sup[xx, yy] == 1:
                                str_B = f"{trans_contacts.chromB}\t{yy * self.selected_resolution}\t{(yy + 1) * self.selected_resolution}"
                                write_str = f"{str_A}\t{str_B}\t1\t{trans_contacts.hic_sum_noartifacts[xx, yy]}\n"
                                chr_contact_bed.write(write_str)
                                all_chr_contact_bed.write(write_str)
                            # add negative label
                            elif trans_contacts.hic_neg_num_sup[xx, yy] == 1:
                                str_B = f"{trans_contacts.chromB}\t{yy * self.selected_resolution}\t{(yy + 1) * self.selected_resolution}"
                                write_str = f"{str_A}\t{str_B}\t0\t{trans_contacts.hic_sum_noartifacts[xx, yy]}\n"
                                chr_contact_bed.write(write_str)
                                all_chr_contact_bed.write(write_str)
                    chr_contact_bed.close()
                    cnt_pos += np.nansum(trans_contacts.hic_pos_num_sup)
                    cnt_neg += np.nansum(trans_contacts.hic_neg_num_sup)

                    trans_contacts.save_obj(save_path_new)
        all_chr_contact_bed.close()
        print(f"cnt_pos: {cnt_pos}")
        print(f"cnt_neg: {cnt_neg}")

    def compute_replicate_metrics(self,
                                  chrom_set,
                                  chrom_set_name="train_chroms"):
        """
        Compute the labels for the replicate group. 
        We will use it to compute the reproducibility upper limit. 
        :param chrom_set: list, list of chromosomes. 
        :param chrom_set_name: str, name of chromosome set. 
        :return: None
        """
        if self.replicate_group == "No":
            # get list of hi-c replicates
            hic_files_testgrp = np.array(
                decode_list(self.config['input_files']['hic_files_testgrp']),
                dtype=str
            )
        else:
            hic_files_testgrp = np.array(
                decode_list(self.config['input_files']['hic_files_repgrp']),
                dtype=str
            )

        hic = hicstraw.HiCFile(hic_files_testgrp[0])
        for chrom in hic.getChromosomes():
            print(chrom.name, chrom.length)
            self.chrom_names.append(chrom.name)
            self.chrom_lengths[chrom.name] = chrom.length

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
                    if self.replicate_group == "No":
                        save_path_sup = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5.gz"
                    else:
                        save_path_sup = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5_replicate.gz"
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
                        hic_values = None

                        for k in range(len(hic_files_testgrp)):

                            # Get Hi-C object
                            hic = hicstraw.HiCFile(hic_files_testgrp[k])

                            chrA_name = chromA  # .replace("chr", "")
                            chrB_name = chromB  # .replace("chr", "")
                            # Get raw Hi-C reads
                            hic_obj = hic.getMatrixZoomData(chromA,
                                                            chromB,
                                                            self.hic_datatype,
                                                            self.hic_norm,
                                                            self.hic_type,
                                                            self.selected_resolution)

                            if hic_values is None:
                                # extract a patch
                                hic_values = hic_obj.getRecordsAsMatrix(startA,
                                                                        endA,
                                                                        startB,
                                                                        endB)
                            else:
                                # extract a patch
                                hic_values += hic_obj.getRecordsAsMatrix(startA,
                                                                         endA,
                                                                         startB,
                                                                         endB)

                        if self.replicate_group == "No":
                            np.savez(f"{self.save_prefix}_{chrom_set_name}_{self.rep_name}_"
                                     f"{self.selected_resolution}_5_v_5_hic_values.npz",
                                     hic_values=hic_values,
                                     hic_pos_num_sup=trans_contacts.hic_pos_num_sup,
                                     hic_neg_num_sup=trans_contacts.hic_neg_num_sup)
                        else:
                            np.savez(f"{self.save_prefix}_{chrom_set_name}_{self.rep_name}_"
                                     f"{self.selected_resolution}_5_v_5_hic_values_replicate.npz",
                                     hic_values=hic_values,
                                     hic_pos_num_sup=trans_contacts.hic_pos_num_sup,
                                     hic_neg_num_sup=trans_contacts.hic_neg_num_sup)

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

        if self.replicate_group == "No":
            np.savez(f"{self.save_prefix}_{chrom_set_name}_{self.rep_name}_"
                     f"{self.selected_resolution}_5_v_5.npz",
                     all_labels=all_labels,
                     all_preds=all_preds)
        else:
            np.savez(f"{self.save_prefix}_{chrom_set_name}_{self.rep_name}_"
                     f"{self.selected_resolution}_5_v_5_replicate.npz",
                     all_labels=all_labels,
                     all_preds=all_preds)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file", type=str, default="config_data_within_tissue.yml", help="path to the config file."
    )

    parser.add_argument(
        "--replicate_group", type=str, default="No", help="Whether we are using the test group or replicate group."
    )

    args = parser.parse_args()

    gw_trans_contact = GenomeWideTransContacts(args.config_file, args.replicate_group)

    gw_trans_contact.extract_gw_trans_contacts()
    gw_trans_contact.compute_gw_artifacts()
    gw_trans_contact.make_gw_supervised_labels()
    gw_trans_contact.compute_replicate_metrics(gw_trans_contact.val_chroms,
                                               chrom_set_name="val_chroms")
    gw_trans_contact.compute_replicate_metrics(gw_trans_contact.test_chroms,
                                               chrom_set_name="test_chroms")
"""
twinc_data.py
Author: Anupama Jha <anupamaj@uw.edu>
In this script, we generate labels for
trans-contacts
"""

import gzip
import hicstraw
import numpy as np
import _pickle as pickle


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
        TransContact class for every chromosome.
        This class keeps track of reads from all
        the replicates.
        :param chromA: str, chromosome A
        :param chromB: str, chromosome B
        :param chromA_len: int, length of chromosome A
        :param chromB_len: int, length of chromosome B
        :param resolution: int, Hi-C data resolution
                                default=100KB
        :param rep_name: str, name of the experiment,
                              usually the tissue/cell line.
        :param hic_files: list, list of replicate hi-c files.
        :param reads_threshold_pos: float, positive read threshold.
        :param reads_threshold_neg: float, negative read threshold.
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
        Extract genomic locus pairs, which have more reads
        than the positive threshold for at least the number
        of required donors as positive labels and the locus
        pairs, which have fewer reads than the negative
        threshold for at least the number of required donors
        as negative labels.
        :param hic_datatype: str, observed or observed/expected
        :param hic_raw: str, if raw Hi-C needed use NONE
        :param hic_norm: str, if norm needed, e.g., KR
        :param hic_type: str, type of Hi-C entry, e.g., BP
        :return: None
        """
        for k in range(len(self.hic_files)):
            startA = 0
            endA = int(self.chromA_len)

            startB = 0
            endB = int(self.chromB_len)

            print(f"startA, endA: {startA, endA}")
            print(f"startB, endB: {startB, endB}")

            print(f"hic: {self.hic_files[k]}, {self.chromA}, {self.chromB}")

            # Get Hi-C object
            hic = hicstraw.HiCFile(self.hic_files[k])

            chrA_name = self.chromA  # .replace("chr", "")
            chrB_name = self.chromB  # .replace("chr", "")

            # Get raw Hi-C reads
            hic_obj_raw = hic.getMatrixZoomData(chrA_name,
                                                chrB_name,
                                                hic_datatype,
                                                hic_raw,
                                                hic_type,
                                                self.resolution)
            # extract a patch
            hic_values_raw = hic_obj_raw.getRecordsAsMatrix(startA,
                                                            endA,
                                                            startB,
                                                            endB)

            print(f"hic_values_raw: {hic_values_raw}")

            print(f"NaN hic_values_raw: {np.sum(np.isnan(hic_values_raw))}")

            # compute at what quantile is the read threshold for positives
            raw_quantile_pos = (hic_values_raw <= self.reads_threshold_pos).mean()
            print(f"Reads threshold for positive label: {self.reads_threshold_pos}")
            print(f"Reads quantile for positive label: {raw_quantile_pos}")

            # compute at what quantile is the read theshold for negatives
            raw_quantile_neg = (hic_values_raw <= self.reads_threshold_neg).mean()
            print(f"Reads threshold for negative label: {self.reads_threshold_neg}")
            print(f"Reads quantile for negative label: {raw_quantile_neg}")

            chrA_name = self.chromA  # .replace("chr", "")
            chrB_name = self.chromB  # .replace("chr", "")

            # Get scale normalized data
            hic_obj = hic.getMatrixZoomData(chrA_name,
                                            chrB_name,
                                            hic_datatype,
                                            hic_norm,
                                            hic_type,
                                            self.resolution)
            # extract a patch
            hic_values = hic_obj.getRecordsAsMatrix(startA,
                                                    endA,
                                                    startB,
                                                    endB)

            print(f"hic_values: {np.sum(np.isnan(hic_values))}")
            print(f"NaN hic_values: {np.sum(np.isnan(hic_values))}")

            # extract the normalized reads corresponding the raw reads
            norm_quantile_pos = np.quantile(hic_values, raw_quantile_pos)
            print(f"Norm Reads quantile for positive label: {norm_quantile_pos}")

            # extract the normalized reads corresponding the raw reads
            norm_quantile_neg = np.quantile(hic_values, raw_quantile_neg)
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
            print(f"self.hic_pos_num: {np.nansum(self.hic_pos_num)}")
            print(f"self.hic_neg_num: {np.nansum(self.hic_neg_num)}")

    def remove_top_bottom_artifacts(self,
                                    top_percentile,
                                    bottom_percentile):
        """
        Remove the top and bottom artifacts by read count.
        :param top_percentile: float, top percentile for removal
        :param bottom_percentile: flaot, bottom percentile for removal
        :return: None
        """
        print("In remove_top_bottom_artifacts")
        self.top_percentile = top_percentile
        self.bottom_percentile = bottom_percentile

        row_artifacts = self.artifact_ranking[self.chromA]
        col_artifacts = self.artifact_ranking[self.chromB]

        self.top_row_artifact_threshold = np.quantile(row_artifacts, self.top_percentile)
        self.bottom_row_artifact_threshold = np.quantile(row_artifacts, self.bottom_percentile)

        self.top_row_artifacts = np.where(row_artifacts >= self.top_row_artifact_threshold)[0]
        self.bottom_row_artifact = np.where(row_artifacts <= self.bottom_row_artifact_threshold)[0]

        self.top_col_artifact_threshold = np.quantile(col_artifacts, self.top_percentile)
        self.bottom_col_artifact_threshold = np.quantile(col_artifacts, self.bottom_percentile)

        self.top_col_artifacts = np.where(col_artifacts >= self.top_col_artifact_threshold)[0]
        self.bottom_col_artifact = np.where(col_artifacts <= self.bottom_col_artifact_threshold)[0]

        self.hic_sum_noartifacts = np.array(self.hic_sum, copy=True)

        self.hic_sum_noartifacts[self.top_row_artifacts, :] = np.nan
        self.hic_sum_noartifacts[self.bottom_row_artifact, :] = np.nan

        self.hic_sum_noartifacts[:, self.top_col_artifacts] = np.nan
        self.hic_sum_noartifacts[:, self.bottom_col_artifact] = np.nan

    def remove_pos_artifacts(self,
                             nansum_thr_pos=0.9,
                             total_notnan_frac_pos=0.9):
        """
        Remove genomic locus with contacts across the entire
        chromosome if the number of contacts is greater than
        the specified threshold, as these are likely to be
        artifacts from Hi-C experiment.
        :param nansum_thr_pos: float, fraction of contacts
        :param total_notnan_frac_pos: float, fraction of
                                      not nan values for
                                      a row/column.
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
        Remove genomic loci that don't interact with other
        regions as these are likely to be repeat region and
        we cannot map to them.
        :param nansum_thr_neg: float, fraction of no contacts
        :param total_notnan_frac_neg: float, fraction of not
                                             nan values.
        :return: None.
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
        Given the object with top, bottom and chromosome level
        artifacts filtered out, we will now write the supervised
        labels.
        :param pos_samples: int, number of positive samples.
        :param neg_samples: int, number of negative samples.
        :param top_percentile: float, top percentile for removal.
        :param bottom_percentile: flaot, bottom percentile for removal.
        :param nansum_thr_pos: float, fraction of contacts for positive
                                      row/column removal at chromosome level.
        :param total_notnan_frac_pos: float, fraction of not nans for
                                     positive row/column removal at chromosome level.
        :param nansum_thr_neg: float, fraction of contacts for negative
                                      row/column removal at chromosome level.
        :param total_notnan_frac_neg: float, fraction of not nans for
                                     positive row/column removal at chromosome level.
        :return: None
        """
        print("In make_supervised_labels")

        self.remove_top_bottom_artifacts(top_percentile, bottom_percentile)
        print(f"self.hic_pos_num: {np.nansum(self.hic_pos_num)}")
        print(f"self.hic_neg_num: {np.nansum(self.hic_neg_num)}")

        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.hic_pos_num_sup = np.where(self.hic_pos_num >= self.pos_samples, 1.0, 0.0)
        print(f"self.hic_pos_num_sup: {np.nansum(self.hic_pos_num_sup)}")

        self.hic_pos_num_sup[self.top_row_artifacts, :] = np.nan
        self.hic_pos_num_sup[self.bottom_row_artifact, :] = np.nan

        self.hic_pos_num_sup[:, self.top_col_artifacts] = np.nan
        self.hic_pos_num_sup[:, self.bottom_col_artifact] = np.nan

        self.hic_neg_num_sup = np.where(self.hic_neg_num >= self.neg_samples, 1.0, 0.0)
        print(f"self.hic_neg_num_sup: {np.nansum(self.hic_neg_num_sup)}")

        self.hic_neg_num_sup[self.top_row_artifacts, :] = np.nan
        self.hic_neg_num_sup[self.bottom_row_artifact, :] = np.nan

        self.hic_neg_num_sup[:, self.top_col_artifacts] = np.nan
        self.hic_neg_num_sup[:, self.bottom_col_artifact] = np.nan

        self.remove_pos_artifacts(nansum_thr_pos, total_notnan_frac_pos)
        print(f"self.hic_pos_num_sup: {np.nansum(self.hic_pos_num_sup)}")
        print(f"self.hic_neg_num_sup: {np.nansum(self.hic_neg_num_sup)}")

        self.remove_neg_artifacts(nansum_thr_neg, total_notnan_frac_neg)
        print(f"self.hic_pos_num_sup: {np.nansum(self.hic_pos_num_sup)}")
        print(f"self.hic_neg_num_sup: {np.nansum(self.hic_neg_num_sup)}")

        self.cnt_pos = np.nansum(self.hic_pos_num_sup)
        self.cnt_neg = np.nansum(self.hic_neg_num_sup)

    def save_obj(self, save_path):
        """
        Save pickle object, one per chromosome pair.
        :param save_path: str, path for saving.
        :return: None.
        """
        with gzip.open(save_path, 'wb') as save_obj:
            pickle.dump(self, save_obj)


class GenomeWideTransContacts():
    """
    Genome wide trans-contact class for managing
    all chromosome pairs.
    """

    def __init__(self,
                 config_file,
                 replicate_group):
        """
        Constructor
        :param config_file: str, path to config file
        :param replicate_group:  str, whether we are
                                      processing the
                                      replicate group.
        """
        # make a config parser object
        self.config = configparser.ConfigParser()

        # read its parameters
        self.config.read(config_file)

        self.replicate_group = replicate_group

        # get the condition (within tissue/cross tissue)
        # in which we are computing reproducible contacts
        self.rep_name = self.config['data_parameters']['rep_name']

        if self.replicate_group == "No":
            # get list of hi-c replicates
            self.hic_files_repgrp = np.array(
                decode_list(self.config['input_files']['hic_files_repgrp']),
                dtype=str
            )
        else:
            self.hic_files_repgrp = np.array(
                decode_list(self.config['input_files']['hic_files_testgrp']),
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
        """
        # get chromosome names and lengths
        hic = hicstraw.HiCFile(self.hic_files_repgrp[0])
        for chrom in hic.getChromosomes():
            print(chrom.name, chrom.length)
            self.chrom_names.append(chrom.name)
            self.chrom_lengths[chrom.name] = chrom.length

        # for all trans chromosome pairs
        for i in range(1, len(self.chroms_order), 1):
            for j in range(i + 1, len(self.chroms_order) + 1, 1):
                chromA = self.chroms_order[i]
                chromB = self.chroms_order[j]
                print(chromA, chromB)
                # If processing label group
                if self.replicate_group == "No":
                    save_path = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5.gz"
                # If processing repliacte group
                else:
                    save_path = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5_replicate.gz"
                # If TransContact object doesn't exist
                if not os.path.exists(save_path):
                    # Make one
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
                    # get reproducible contacts
                    trans_contacts.extract_reproducible_contacts(hic_datatype=self.hic_datatype,
                                                                 hic_norm=self.hic_norm,
                                                                 hic_raw=self.hic_raw,
                                                                 hic_type=self.hic_type)
                    # save it
                    trans_contacts.save_obj(save_path)
                else:
                    print(f"{save_path} exist, delete it to rerun.")

    def compute_gw_artifacts(self):
        """
        Compute top and bottom artifacts based on total genome-wide contact count for genomic locus.
        :return: None
        """
        # for all chromosomes
        for i in range(1, len(self.chroms_order) + 1, 1):
            chromA = self.chroms_order[i]
            flipped = False
            base_chom_rows = None
            for j in range(1, len(self.chroms_order) + 1, 1):
                chromB = self.chroms_order[j]
                # If trans-chromosome pair
                if chromA != chromB:
                    # If processing label group
                    if self.replicate_group == "No":
                        save_path = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5.gz"
                    # If processing replicate group
                    else:
                        save_path = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5_replicate.gz"
                    # If TransContact object doesn't exist.
                    if not os.path.exists(save_path):
                        if self.replicate_group == "No":
                            save_path = f"{self.save_prefix}/{chromB}_{chromA}_{self.rep_name}_{self.selected_resolution}_5_v_5.gz"
                        else:
                            save_path = f"{self.save_prefix}/{chromB}_{chromA}_{self.rep_name}_{self.selected_resolution}_5_v_5_replicate.gz"
                        # If chromB < chromA in the ordering, we will need to flip it.
                        flipped = True
                    # Open the saved object
                    with gzip.open(save_path, 'rb') as save_obj:
                        print(save_path)
                        trans_contacts = pickle.load(save_obj)
                        # If flipping needed
                        if flipped:
                            flipped = False
                            # If first chrom pair and flipping needed
                            if base_chom_rows is None:
                                # flip
                                base_chom_rows = trans_contacts.hic_sum.T
                                print(f"flipped: base_chom_rows: {base_chom_rows.shape}")
                            # If not first chrom pair, but flipping needed
                            else:
                                print(f"flipped: trans_contacts.hic_sum.T: {trans_contacts.hic_sum.T.shape}")
                                base_chom_rows = np.concatenate((base_chom_rows,
                                                                 trans_contacts.hic_sum.T), axis=1)
                            print(f"flipped: base_chom_rows: {base_chom_rows.shape}")
                        else:
                            # If first chrom pair but no flipping needed
                            if base_chom_rows is None:
                                base_chom_rows = trans_contacts.hic_sum
                                print(f"Not flipped: base_chom_rows: {base_chom_rows.shape}")
                            # If not first chrom pair but no flipping needed
                            else:
                                print(f"Not flipped trans_contacts.hic_sum: {trans_contacts.hic_sum.shape}")
                                base_chom_rows = np.concatenate((base_chom_rows,
                                                                 trans_contacts.hic_sum), axis=1)
                            print(f"Not flipped: base_chom_rows: {base_chom_rows.shape}")

            # get sum along first axis
            base_chom_sum = np.sum(base_chom_rows, axis=1)
            print(f"base_chom_sum: {base_chom_sum.shape}")

            # add sum to the artifact ranking
            self.artifact_ranking[chromA] = base_chom_sum

        # for all chromosome pairs, add artifact ranking
        # to the TransContact object.
        for i in range(1, len(self.chroms_order), 1):
            for j in range(i + 1, len(self.chroms_order) + 1, 1):
                chromA = self.chroms_order[i]
                chromB = self.chroms_order[j]
                if self.replicate_group == "No":
                    save_path = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5.gz"
                else:
                    save_path = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5_replicate.gz"

                with gzip.open(save_path, 'rb') as save_obj:
                    print(chromA, chromB)
                    trans_contacts = pickle.load(save_obj)

                    trans_contacts.artifact_ranking[chromA] = self.artifact_ranking[chromA]
                    trans_contacts.artifact_ranking[chromB] = self.artifact_ranking[chromB]

                    trans_contacts.save_obj(save_path)

    def make_gw_supervised_labels(self):
        """
        Make supervised labels for all trans-chromosome pairs..
        :return: None
        """
        cnt_pos = 0
        cnt_neg = 0
        # If label group
        if self.replicate_group == "No":
            all_chr_contact_bed = open(
                f"{self.labels_file}_all_{self.rep_name}_{self.selected_resolution}_5_v_5_sup.txt", 'w')
        # If replicate group
        else:
            all_chr_contact_bed = open(
                f"{self.labels_file}_all_{self.rep_name}_{self.selected_resolution}_5_v_5_sup_replicate.txt", 'w')
        # For all chromosome pairs
        for i in range(1, len(self.chroms_order), 1):
            for j in range(i + 1, len(self.chroms_order) + 1, 1):
                chromA = self.chroms_order[i]
                chromB = self.chroms_order[j]
                print(chromA, chromB)
                # If label group is being processed
                if self.replicate_group == "No":
                    save_path_new = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5.gz"
                # If replicate group is being processed
                else:
                    save_path_new = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5_replicate.gz"

                # open the trans contact object
                with gzip.open(save_path_new, 'rb') as save_obj:
                    trans_contacts = pickle.load(save_obj)
                    # make supervised labels
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
                    # Make label text file for label group
                    if self.replicate_group == "No":
                        chr_contact_bed = open(
                            f"{self.labels_file}_{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5_sup.txt",
                            'w')
                    # Make label text file for replicate group
                    else:
                        chr_contact_bed = open(
                            f"{self.labels_file}_{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5_sup_replicate.txt",
                            'w')
                    # For every locus pair
                    for xx in range(trans_contacts.hic_pos_num_sup.shape[0]):
                        str_A = f"{trans_contacts.chromA}\t{xx * self.selected_resolution}\t{(xx + 1) * self.selected_resolution}"
                        for yy in range(trans_contacts.hic_pos_num_sup.shape[1]):
                            # add positive label
                            if trans_contacts.hic_pos_num_sup[xx, yy] == 1:
                                str_B = f"{trans_contacts.chromB}\t{yy * self.selected_resolution}\t{(yy + 1) * self.selected_resolution}"
                                write_str = f"{str_A}\t{str_B}\t1\t{trans_contacts.hic_sum_noartifacts[xx, yy]}\n"
                                chr_contact_bed.write(write_str)
                                all_chr_contact_bed.write(write_str)
                            # add negative label
                            elif trans_contacts.hic_neg_num_sup[xx, yy] == 1:
                                str_B = f"{trans_contacts.chromB}\t{yy * self.selected_resolution}\t{(yy + 1) * self.selected_resolution}"
                                write_str = f"{str_A}\t{str_B}\t0\t{trans_contacts.hic_sum_noartifacts[xx, yy]}\n"
                                chr_contact_bed.write(write_str)
                                all_chr_contact_bed.write(write_str)
                    chr_contact_bed.close()
                    cnt_pos += np.nansum(trans_contacts.hic_pos_num_sup)
                    cnt_neg += np.nansum(trans_contacts.hic_neg_num_sup)

                    trans_contacts.save_obj(save_path_new)
        all_chr_contact_bed.close()
        print(f"cnt_pos: {cnt_pos}")
        print(f"cnt_neg: {cnt_neg}")

    def compute_replicate_metrics(self,
                                  chrom_set,
                                  chrom_set_name="train_chroms"):
        """
        Compute the labels for the replicate group. 
        We will use it to compute the reproducibility upper limit. 
        :param chrom_set: list, list of chromosomes. 
        :param chrom_set_name: str, name of chromosome set. 
        :return: None
        """
        if self.replicate_group == "No":
            # get list of hi-c replicates
            hic_files_testgrp = np.array(
                decode_list(self.config['input_files']['hic_files_testgrp']),
                dtype=str
            )
        else:
            hic_files_testgrp = np.array(
                decode_list(self.config['input_files']['hic_files_repgrp']),
                dtype=str
            )

        hic = hicstraw.HiCFile(hic_files_testgrp[0])
        for chrom in hic.getChromosomes():
            print(chrom.name, chrom.length)
            self.chrom_names.append(chrom.name)
            self.chrom_lengths[chrom.name] = chrom.length

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
                    if self.replicate_group == "No":
                        save_path_sup = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5.gz"
                    else:
                        save_path_sup = f"{self.save_prefix}/{chromA}_{chromB}_{self.rep_name}_{self.selected_resolution}_5_v_5_replicate.gz"
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
                        hic_values = None

                        for k in range(len(hic_files_testgrp)):

                            # Get Hi-C object
                            hic = hicstraw.HiCFile(hic_files_testgrp[k])

                            chrA_name = chromA  # .replace("chr", "")
                            chrB_name = chromB  # .replace("chr", "")
                            # Get raw Hi-C reads
                            hic_obj = hic.getMatrixZoomData(chromA,
                                                            chromB,
                                                            self.hic_datatype,
                                                            self.hic_norm,
                                                            self.hic_type,
                                                            self.selected_resolution)

                            if hic_values is None:
                                # extract a patch
                                hic_values = hic_obj.getRecordsAsMatrix(startA,
                                                                        endA,
                                                                        startB,
                                                                        endB)
                            else:
                                # extract a patch
                                hic_values += hic_obj.getRecordsAsMatrix(startA,
                                                                         endA,
                                                                         startB,
                                                                         endB)

                        if self.replicate_group == "No":
                            np.savez(f"{self.save_prefix}_{chrom_set_name}_{self.rep_name}_"
                                     f"{self.selected_resolution}_5_v_5_hic_values.npz",
                                     hic_values=hic_values,
                                     hic_pos_num_sup=trans_contacts.hic_pos_num_sup,
                                     hic_neg_num_sup=trans_contacts.hic_neg_num_sup)
                        else:
                            np.savez(f"{self.save_prefix}_{chrom_set_name}_{self.rep_name}_"
                                     f"{self.selected_resolution}_5_v_5_hic_values_replicate.npz",
                                     hic_values=hic_values,
                                     hic_pos_num_sup=trans_contacts.hic_pos_num_sup,
                                     hic_neg_num_sup=trans_contacts.hic_neg_num_sup)

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

        if self.replicate_group == "No":
            np.savez(f"{self.save_prefix}_{chrom_set_name}_{self.rep_name}_"
                     f"{self.selected_resolution}_5_v_5.npz",
                     all_labels=all_labels,
                     all_preds=all_preds)
        else:
            np.savez(f"{self.save_prefix}_{chrom_set_name}_{self.rep_name}_"
                     f"{self.selected_resolution}_5_v_5_replicate.npz",
                     all_labels=all_labels,
                     all_preds=all_preds)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file", type=str, default="config_data_within_tissue.yml", help="path to the config file."
    )

    parser.add_argument(
        "--replicate_group", type=str, default="No", help="Whether we are using the test group or replicate group."
    )

    args = parser.parse_args()

    gw_trans_contact = GenomeWideTransContacts(args.config_file, args.replicate_group)

    gw_trans_contact.extract_gw_trans_contacts()
    gw_trans_contact.compute_gw_artifacts()
    gw_trans_contact.make_gw_supervised_labels()
    gw_trans_contact.compute_replicate_metrics(gw_trans_contact.val_chroms,
                                               chrom_set_name="val_chroms")
    gw_trans_contact.compute_replicate_metrics(gw_trans_contact.test_chroms,
                                               chrom_set_name="test_chroms")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
