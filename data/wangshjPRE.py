#python -B Wangshj.py -snv onesnv,chr12,25398284,C,T,0.5:onesnv,chr12,25398284,C,A,0.5:
# !/usr/bin/env python
# -*- coding:utf-8 -*-
# coding=utf-8
import sys
import os
# #1.0:work on all chr
# path = "E:/ESM.csv"
# str1 = "python -B Wangshj.py -snv "
# f = open(path)
# fout = open("output.sh", "w")
# fout.write(str(str1))
# # lines = f.readline()
# lines = f.readlines()
# line_str2 = " -delete "
# for line in lines:
#     line_arr = line.split(",")
#     if line_arr[3] == "INS":
#         line_str = "oneinsert,"+ str(line_arr[0]) + "," + str(int(line_arr[1])-1) + ",A,A"+ str(line_arr[6]).rstrip("\n") + ",0.5:"
#         fout.write(str(line_str))
#     elif line_arr[3] == "SNP":
#         line_str = "onesnv,"+ str(line_arr[0]) + "," + str(line_arr[1]) + "," + str(line_arr[4]) + "," +  str(line_arr[5]).rstrip("\n") + ",0.5:"
#         fout.write(str(line_str))
#     elif line_arr[3] == "DEL":
#         line_str2 = line_str2 + "onedelete," + str(line_arr[0]) + "," + str(line_arr[1]) + "," + str(line_arr[2]) + ",0.5,0:"
#     # fout = open("features", "w")
#     # fout.write(str(line_str))
# fout.write(str(line_str2))
# fout.close()


#2.0:work on single chr
path = "E:/ESM.csv"
str1 = ""
f = open(path)
fout = open("chrY.sh", "w")
fout.write(str(str1))
# lines = f.readline()
lines = f.readlines()
line_str2 = " -delete "
for line in lines:
    line_arr = line.split(",")
    if line_arr[0] == "chrY":
        # if line_arr[3] == "INS":
        #     line_str = "oneinsert,"+ str(line_arr[0]) + "," + str(int(line_arr[1])-1) + ",A,A"+ str(line_arr[6]).rstrip("\n") + ",0.5:"
        #     fout.write(str(line_str))
        if line_arr[3] == "SNP":
            line_str = "onesnv,"+ str(line_arr[0]) + "," + str(line_arr[1]) + "," + str(line_arr[4]) + "," +  str(line_arr[5]).rstrip("\n") + ",0.5:"
            fout.write(str(line_str))
            line_str = "onesnv," + str(line_arr[0]) + "," + str(line_arr[1]) + "," + str(line_arr[4]) + "," + str(line_arr[6]).rstrip("\n") + ",0.5:"
            fout.write(str(line_str))
        # elif line_arr[3] == "DEL":
        #     line_str2 = line_str2 + "onedelete," + str(line_arr[0]) + "," + str(line_arr[1]) + "," + str(line_arr[2]) + ",0.5,0:"

# fout.write(str(line_str2))
# fout.write(" -out chr1 &")
fout.close()