# Coding: utf8
# -------------------------------------------------------------------------
# preprocess.py
# Some of the operations used in the preprocessing.
# -------------------------------------------------------------------------

import random


def get_cri_pos(filename, target):
    c_pos = []
    f = open(filename, "r", encoding="utf8")
    temp_list = f.readline().split(":")
    for info in temp_list:
        if info:
            info_list = info.split(",")
            if info_list[3] != info_list[4] and info_list[1] == target:
                c_pos.append(info_list[2])
    c_pos = list(set(c_pos))
    return c_pos


def get_ms_information(filename, sig_num):
    ms_information = []
    sig_f = open(filename, "r", encoding="utf8")
    sig_f.readline()
    line = sig_f.readline()
    while line:
        line_list = str(line).split("\t")
        prob_list = []
        for sig_no in range(sig_num):
            prob_list.append(float(line_list[sig_no + 3]))
        prob = max(prob_list)
        base_a = line_list[1]
        base_b = line_list[2][0] + line_list[2][4] + line_list[2][6]
        ms_information.append((base_a, base_b, prob))
        line = sig_f.readline()
    return ms_information


def get_ref_data(r_filename, flag1, flag2=">"):
    chr_ref_data = []
    r_f = open(r_filename, "r", encoding="utf8")
    line = r_f.readline()
    while line:
        if line.find(flag1) != -1:
            line = r_f.readline()
            while line and line.find(flag2) == -1:
                chr_ref_data.append(line.replace("\n", ""))
                line = r_f.readline()
            break
        line = r_f.readline()
    r_f.close()
    return chr_ref_data


def read_ref(ref, start_pos, end_pos):
    if not ref:
        return ""
    start_pos -= 1
    end_pos -= 1
    len_line = len(ref[0])
    ignore_num = start_pos // len_line
    read_num = end_pos // len_line + 1 - ignore_num
    ref_frag = ""
    for i in range(ignore_num, ignore_num + read_num):
        ref_frag += ref[i]
    start_pos = start_pos - ignore_num * len_line
    end_pos = end_pos - ignore_num * len_line
    return ref_frag[start_pos: end_pos].upper()


def trans2num(base):
    temp_result = []
    for base_point in base:
        if base_point == "A":
            temp_result.append(1)
        elif base_point == "T":
            temp_result.append(2)
        elif base_point == "C":
            temp_result.append(4)
        elif base_point == "G":
            temp_result.append(8)
        else:
            return False
    return temp_result


def generate_dataset(sam_filename, save_filename, cri_list, ms_information, ref, target, ignore_line=26,
                     base_length=75):
    sam_file = open(sam_filename, "r", encoding="utf8")
    save_file = open(save_filename, "w", encoding="utf8")
    for ln in range(ignore_line):
        sam_file.readline()
    line = sam_file.readline()
    while line:
        line_list = str(line).split("\t")
        base_column = line_list[9].upper()
        base_len = len(base_column)
        if base_len == base_length and line_list[2] == target:
            # Get input.
            sam_pos_1 = int(line_list[3])
            sam_pos_2 = sam_pos_1 + base_len
            temp_result = []
            temp_base = trans2num(base_column)
            ref_base = read_ref(ref, sam_pos_1, sam_pos_2)
            temp_ref = trans2num(ref_base)
            if temp_base is False or temp_ref is False:
                line = sam_file.readline()
                continue
            for i in range(base_len):
                temp_result.append(temp_base[i] - temp_ref[i])
            temp = []
            for i in range(base_len):
                if temp_result[i] != 0:
                    temp.append(i)
            mutation_information = []
            for i in temp:
                mutation_information.append((ref_base[i], base_column[i], i + sam_pos_1))
                if i - 1 not in temp and i - 1 >= 0:
                    temp_result[i - 1] = trans2num(base_column[i - 1])[0]
                if i + 1 not in temp and i + 1 < base_len:
                    temp_result[i + 1] = trans2num(base_column[i + 1])[0]
            # Get label.
            label = 0
            if temp:
                for pos in cri_list:
                    if sam_pos_1 <= int(pos) < sam_pos_2:
                        label = 1
                        break
            # Get MS information.
            prob = add_signature_prob(base_column, ref, ms_information, sam_pos_1, sam_pos_2)
            # Make samples
            save_file.write(
                str(temp_result) + "\t" + str(prob) + "\t" + str(label) + "\t" + str(mutation_information) + "\n")
        line = sam_file.readline()
    sam_file.close()
    save_file.close()


def add_signature_prob(base, ref, sig_pairs, pos_1, pos_2):
    prob = 0
    base_ref = read_ref(ref, pos_1, pos_2)
    for sig in sig_pairs:
        ft_pos = base.find(sig[1])
        if ft_pos != -1 and base_ref[ft_pos: ft_pos + len(sig[0])] == sig[0] and sig[2] > prob:
            prob = sig[2]
    return prob


def classify_dataset(filename):
    f = open(filename, "r", encoding="utf8")
    filename_1 = filename.split(".")[0] + "_0_0.sam"
    filename_2 = filename.split(".")[0] + "_0_1.sam"
    filename_3 = filename.split(".")[0] + "_ms_0.sam"
    filename_4 = filename.split(".")[0] + "_ms_1.sam"
    f1 = open(filename_1, "w", encoding="utf8")
    f2 = open(filename_2, "w", encoding="utf8")
    f3 = open(filename_3, "w", encoding="utf8")
    f4 = open(filename_4, "w", encoding="utf8")
    line = f.readline()
    while line:
        line_list = line.split("\t")
        if line_list[1] == "0":
            if line_list[2].find("0") != -1:
                f1.write(line)
            else:
                f2.write(line)
        else:
            if line_list[2].find("0") != -1:
                f3.write(line)
            else:
                f4.write(line)
        line = f.readline()
    f.close()
    f1.close()
    f2.close()
    f3.close()
    f4.close()


def make_samples(filename_list, count_list, prob, sample_filename):
    save_f = open(sample_filename, "w", encoding="utf8")
    upper = len(filename_list)
    if upper != len(count_list):
        return
    for i in range(upper):
        count = count_list[i]
        while count > 0:
            f = open(filename_list[i], "r", encoding="utf8")
            line = f.readline()
            while line and count > 0:
                if random.random() < prob:
                    save_f.write(line)
                    count -= 1
                line = f.readline()
            f.close()


if __name__ == "__main__":
    # ---------- Parameter ---------- #
    chrom = "chr2"
    prob_num = 30
    cri_filename = "data/source/chr2.sh"
    ms_filename = "data/source/signatures_probabilities.txt"
    ref_filename = "data/source/hg19.fa"
    read_filename = "data/source/chr2_1x.sam"
    dataset_filename = "data/processed/dateset_chr2_1x.sam"
    # ---------- Parameter ---------- #
    
    # cri_pos_list = get_cri_pos(cri_filename, chrom)
    # base_pairs = get_ms_information(ms_filename, prob_num)
    # ref_data = get_ref_data(ref_filename, chrom)
    # generate_dataset(read_filename, dataset_filename, cri_pos_list, base_pairs, ref_data, chrom)

    classify_dataset(dataset_filename)
    make_samples([dataset_filename], [20000], 0.1, "data/processed/sample.sam")
