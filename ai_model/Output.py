from .net import *
import numpy as np
import pandas as pd
import math
import os


def round2(lis):
    for i in range(len(lis)):
        lis[i] = round(lis[i] * 100, 2)
    return lis


class Output_prob():
    def __init__(self, allMeta):
        # super.__init__()
        self.allmeta = allMeta

    def test_m(self, model_pth, spectra, peaks,cls):
        nmrf = NMRformer(
            input_dim=128,
            num_classes=cls,
            dim=256,
            mlp_dim=512
        )
        nmrf.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))
        nmrf = nmrf.eval()
        sorted_peak = sorted(peaks)
        out_csv = pd.DataFrame({'peak': sorted_peak})
        peak_emb = []
        # n_peaks=[]
        n_pos_peak = []
        con = np.ones((len(sorted_peak), len(sorted_peak)))
        for j in range(len(sorted_peak)):
            # n_peaks.append(sorted_peak[j])
            sub_spe = np.array(spectra[int(sorted_peak[j] * 5000) - 64:int(sorted_peak[j] * 5000) + 64])
            peak = (sorted_peak[j] / 12) * 100
            peak_emb.append(sub_spe)
            n_pos_peak.append(peak)
            for ci in range(j + 1, len(sorted_peak)):
                conc = spectra[int(sorted_peak[j] * 5000)] / spectra[int(sorted_peak[ci] * 5000)]
                if conc > 1: conc = 1 / conc
                con[j, ci] = conc
                con[ci, j] = conc
        peak_emb = np.array(peak_emb)
        # n_peaks=np.array(n_peaks)
        n_pos_peak = np.array(n_pos_peak)
        pe = np.zeros((len(n_pos_peak), 128))
        div_term = np.exp(np.arange(0, 128, 2) * (-math.log(10000) / 128))
        pe[:, 0::2] = np.sin(n_pos_peak[:, None] * div_term[None, :])
        pe[:, 1::2] = np.cos(n_pos_peak[:, None] * div_term[None, :])
        # print(peak_emb.shape)
        # print(pe.shape)
        peak_emb = peak_emb + pe
        input = torch.Tensor(peak_emb).unsqueeze(0)
        con = torch.Tensor(con).unsqueeze(0)
        # con_pos=torch.Tensor(con_pos).unsqueeze(0)
        output1 = nmrf(input, con)
        prob = torch.softmax(output1[0], dim=1)
        valuesp, indicesp = torch.topk(prob, k=3, dim=1)

        out_csv['指认结果'] = self.indices_to_meta(indicesp[0][0].tolist())
        out_csv['概率%'] = round2(valuesp[0][0].tolist())
        out_csv['候选结果'] = self.indices_to_meta(indicesp[0][1].tolist())
        out_csv['候选概率%'] = round2(valuesp[0][1].tolist())
        out_csv['peak'] = out_csv['peak'].apply(lambda x: round(x, 4))
        # out_csv['pred_2'] = self.indices_to_meta(indicesp[0][2].tolist())
        # out_csv['prob_2'] = valuesp[0][2].tolist()

        return out_csv

    def indices_to_meta(self, indices):
        meta = []
        for i in indices:
            meta.append(self.allmeta[i - 1])
        return meta
