# Coding: utf8
# -------------------------------------------------------------------------
# textCNN.py
# Classification model based on TextCNN.
# Training & testing process and generate VCF result.
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import math
from sklearn.metrics import roc_auc_score

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Parameter ---------- #
# Data Parameter
train_filename = "train_sample.sam"
test_filename = "test_sample.sam"
model_filename = "model.pth"
vcf_filename = "result.vcf"
chrom = "1"
# TextCNN Parameter
embedding_size = 5
kernel_size = [2, 3, 4]
output_channel = 2
num_classes = 2
ms_coeff = 0
# Train&Test Parameter
training_times = 2000
# ---------- Parameter ---------- #


def read_file(filename):
    samples = []
    probs = []
    labels = []
    information = []
    read_f = open(filename, "r", encoding="utf8")
    line = read_f.readline().replace("\n", "")
    while line:
        line_list = str(line).split("\t")
        sample = eval(line_list[0])
        upr = len(sample)
        for base in range(upr):
            sample[base] += 7  # For embedding step
        samples.append(sample)
        probs.append(eval(line_list[1]))
        labels.append(eval(line_list[2]))
        information.append(eval(line_list[3]))
        line = read_f.readline().replace("\n", "")
    return samples, probs, labels, information


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=20,
                                      embedding_dim=embedding_size)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=in_channel,
                                    out_channels=output_channel,
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=embedding_size - h + 1))
            for h in kernel_size
        ])
        self.fc = nn.Linear(output_channel * len(kernel_size) + 1, num_classes)

    def forward(self, x, x_prob):
        x = self.embedding(x)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = torch.cat([out, x_prob], dim=1)
        out = out.view(-1, out.size(1))
        out = self.fc(out)
        return out


# ---------- Input ---------- #
input_batch, prob_batch, target_batch, _ = read_file(train_filename)
batch_size = len(input_batch)
in_channel = len(input_batch[0])
input_batch, target_batch = torch.LongTensor(input_batch), torch.LongTensor(target_batch)
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset, batch_size, True)
prob_upper = len(prob_batch)
for i in range(prob_upper):
    prob_batch[i] *= ms_coeff
prob_batch = torch.tensor(prob_batch).unsqueeze(1).unsqueeze(1)
# ---------- Input ---------- #


# ---------- Model ---------- #
model = TextCNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# ---------- Model ---------- #


# ---------- Train ---------- #
for epoch in range(training_times):
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x.long(), prob_batch)
        loss = criterion(pred, batch_y)
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            torch.save(model.state_dict(), model_filename)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# ---------- Train ---------- #


# ---------- Test ---------- #
test_samples, test_probs, test_labels, mutation_inf = read_file(test_filename)
FP, TP, FN, TN = 0, 0, 0, 0
result_prob = []
vcf = open(vcf_filename, "a", encoding="utf8")
vcf.write("#CHROM\tPOS\tID\tREF\tALT\n")
upper = len(test_samples)
for i in range(upper):
    test = torch.LongTensor(test_samples[i]).to(device).unsqueeze(0)
    prob = torch.tensor([test_probs[i] * ms_coeff]).to(device).unsqueeze(1).unsqueeze(0)
    model = TextCNN().to(device)
    model.load_state_dict(torch.load(model_filename))
    model = model.eval()
    predict = model(test.long(), prob)
    result_prob.append(float(1 / (1 + math.exp(-predict[0][1] + predict[0][0]))))
    predict = predict.data.max(1, keepdim=True)[1]
    temp_p = int(predict[0][0])
    temp_l = int(test_labels[i])
    if temp_p == temp_l == 1:
        TP += 1
    elif temp_p == temp_l == 0:
        TN += 1
    elif temp_p == 1 and temp_l == 0:
        FP += 1
    else:
        FN += 1
    # Generate VCF
    if temp_p == 1:
        for inf in mutation_inf:
            vcf.write("{0}\t{1}\t.\t{2}\t{3}\n".format(chrom, inf[2], inf[0], inf[1]))
vcf.close()
accuracy = (TP + TN) / (TP + TN + FP + FN)
print("TP:   " + str(TP))
print("FP:   " + str(FP))
print("TN:   " + str(TN))
print("FN:   " + str(FN))
print("ACC:  " + str(accuracy))
if len(set(test_labels)) > 1:
    print("AUC:  " + str(roc_auc_score(test_labels, result_prob)))
# ---------- Test ---------- #
