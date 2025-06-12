import nmrglue as ng

import ai_model.funcs

import numpy as np
from scipy import interpolate, optimize


def baseline_fitting(x, y, start_idx, end_idx):
    """
    对指定区间的数据点进行多项式拟合，返回拟合的基线。
    """
    # 选择第1000到2000下标的x和y
    x_range = x[start_idx:end_idx]
    y_range = y[start_idx:end_idx]

    # 使用多项式拟合基线（这里用二次多项式）
    coeffs = np.polyfit(x_range, y_range, 2)  # 二次多项式拟合
    baseline = np.polyval(coeffs, x_range)  # 计算拟合的基线

    return baseline


def data_process(file_path):
    dic, data = ng.bruker.read(file_path)
    datap = ng.proc_base.em(data, lb=1 / dic["acqus"]["SW_h"])
    datap = ng.proc_base.fft(datap)
    datap = ng.bruker.remove_digital_filter(dic, datap, post_proc=True)
    datap = ng.proc_autophase.autops(datap, fn='acme', disp=False)

    datap = ng.proc_base.rev(datap)
    udic = ng.bruker.guess_udic(dic, datap, strip_fake=True)
    uc = {i: ng.fileiobase.uc_from_udic(udic, dim=i) for i in range(udic["ndim"])}
    x = uc[0].ppm_scale()[::-1]
    y = datap.real[::-1]

    s = 0
    e = len(x) - 1
    for i in range(len(x)):
        if x[i] > -1:
            s = i
            break
    for i in range(len(x)):
        if x[e - i] < 12:
            e = e - i
            break
    x = x[s:e + 1]
    y = y[s:e + 1]



    # 对谱图进行插值
    x_1 = np.linspace(-1, 12, 65000)
    f = interpolate.interp1d(x, y, kind='cubic', bounds_error=False, fill_value=0)
    y_1 = f(x_1)

    y_1 = ai_model.funcs.find_tsp(y_1.tolist())
    # 在第1000到2000下标进行基线拟合
    baseline = baseline_fitting(x_1, y_1, 1000, 2000)

    # 将拟合的基线从谱图中减去
    y_corrected = y_1 - np.min(baseline)*np.ones(len(y_1))  # 保持y数组长度一致

    return y_corrected


if __name__ == '__main__':
    print(data_process(r'E:\code\NMR_data\细胞论文数据\F\10'))