from settings import *
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np


class BilstmCRFModel(nn.Module):

    def __init__(self, config, emb_mat: np.array, freeze: bool):
        """
        :param config: 模型的配置
        :param emb_mat: 词向量编码的矩阵
        :param freeze: 词向量的编码方式，在训练过程中能否调整。0为可以调整，1为不能调整
        """
        super(BilstmCRFModel, self).__init__()
        self.config = config
        # self.embedding = nn.Embedding(dict_size, config.embedding_size)
        emb_matrix = torch.FloatTensor(emb_mat)
        self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=freeze)
        self.bilstm = nn.LSTM(config.embedding_size, config.hidden_size, bidirectional=True, batch_first=True)
        self.emit = nn.Linear(2 * config.hidden_size, OUTPUT_DIC_SIZE)
        self.trans = nn.Parameter(torch.ones(OUTPUT_DIC_SIZE, OUTPUT_DIC_SIZE) * 1 / OUTPUT_DIC_SIZE)

    def forward(self, src, lengths):
        embedding = self.embedding(src)
        packed = pack_padded_sequence(embedding, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        emit_score = self.emit(rnn_out)
        # emit_score = self.dropout(emit_score)
        crf_scores = emit_score.unsqueeze(2).expand(-1, -1, OUTPUT_DIC_SIZE, -1) + self.trans.unsqueeze(0)
        return crf_scores

    def calc_loss(self, scores, targets):
        # batch_size, max_len = targets.size()
        max_len = scores.shape[1]
        targets = targets[:, :max_len]
        mask = (targets != PAD_IDX)
        lengths = mask.sum(dim=1)
        # 将标注好的正确结果转变为索引值的形式，便于后续分数计算。
        # 例如，如果上一状态为3，当前状态为4，输出的可能值有7个（OUTPUT_SIZE=7），那么在计算分数时，就要去找预测结果中3*7+4这个位置的分数。
        for col in range(max_len - 1, 0, -1):
            targets[:, col] += (targets[:, col - 1] * OUTPUT_DIC_SIZE)
        targets[:, 0] += (BOS_IDX * OUTPUT_DIC_SIZE)  # 第一项之前没有“上一个状态”，因此认为上一个状态为BOS_IDX。以此来计算第一项对应的索引值。
        # 计算标注好的正确结果，在使用当前参数下的路径分数
        masked_targets = targets.masked_select(mask)  # 这个函数是为了使标注值为PAD_IDX的项不参与训练。把它们从原始数组中移除。
        masked_scores = scores.masked_select(mask.view(self.config.batch_size, max_len, 1, 1).expand_as(scores)).view(-1, OUTPUT_DIC_SIZE * OUTPUT_DIC_SIZE).contiguous()
        golden_scores = masked_scores.gather(dim=1, index=masked_targets.unsqueeze(1)).sum()  # 针对每个生成的分数列表，根据正确的路径对应的索引值，找到正确路径对应的分数，并将他们求和
        # 计算当前参数下的所有路径分数的总和
        # scores_upto_t[i, j]表示第i个句子的第t个词被标注为j标记的所有t时刻事前的所有子路径的分数之和
        scores_upto_t = torch.zeros(self.config.batch_size, OUTPUT_DIC_SIZE)
        for t in range(max_len):
            # 当前时刻 有效的batch_size（因为有些序列比较短)
            batch_size_t = (lengths > t).sum().item()
            if t == 0:
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, BOS_IDX, :]
            else:
                scores_upto_t[:batch_size_t] = torch.logsumexp(
                    scores[:batch_size_t, t, :, :] +
                    scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1
                )
        all_path_scores = scores_upto_t[:, EOS_IDX].sum()

        # 训练大约两个epoch loss变成负数，从数学的角度上来说，loss = -logP
        loss = (all_path_scores - golden_scores) / self.config.batch_size
        return loss

    def predict(self, src, lengths):
        """使用viterbi方法，针对测试用的输入数据，给出输出数据"""
        crf_scores = self.forward(src, lengths)
        max_len = crf_scores.shape[1]
        score_by_step = torch.zeros(max_len, OUTPUT_DIC_SIZE)
        state_seq = torch.zeros((max_len, OUTPUT_DIC_SIZE)).long()  # 记录前一状态到后一状态的对应关系。该对应关系可以使转移概率最大。
        # 通过向前推导，计算每个环节中，各个状态下可能的最大分数，以及为了达到这个分数，前一个状态应当是什么。
        score_by_step[0, :] = crf_scores[0, 0, BOS_IDX, :]
        for step in range(1, max_len):
            cur_scores = score_by_step[step - 1, :].unsqueeze(1) + crf_scores[0, step, :, :]
            max_scores, prev_states = torch.max(cur_scores, dim=0)
            score_by_step[step, :] = max_scores
            state_seq[step - 1, :] = prev_states
        reversed_states = [torch.argmax(score_by_step[max_len - 1, :]).item()]
        # 向后推导，找到每一个步骤的状态。
        for step in range(max_len - 2, -1, -1):
            reversed_states.append(state_seq[step, reversed_states[-1]].item())
        return list(reversed(reversed_states))
