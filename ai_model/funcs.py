import plotly.graph_objs as go
import pywt
from scipy.signal import find_peaks

from .Output import *


def get_allmeta(path):
    all_meta = []
    with open(path) as f:
        lines = f.readlines()
        for k in lines:
            all_meta.append(k.strip())
    return all_meta


def indices_to_meta(indices, all_meta):
    meta = []
    for i in indices:
        meta.append(all_meta[i - 1])
    return meta


def read_floats_from_txt_files():
    all_floats = []
    with open('./ai_model/alp.txt', 'r') as f:
        for line in f:
            all_floats.append(float(line.strip()))
    return all_floats


def get_peaks(y):
    # 使用小波变换
    widths = np.arange(1, 10)
    x = np.linspace(0, 12, 60000)
    cwt_matrix = pywt.cwt(y, widths, wavelet='mexh')[0]

    # 检测峰值
    peaks = find_peaks(cwt_matrix[5], height=0.005)[0]
    return x[peaks]


def find_tsp(data):
    tmp = data
    m = max(tmp[0:5100])
    idx = tmp.index(m)
    # tmp = [x/m for x in tmp]
    if idx > 5000:
        tmp = tmp[idx - 5000:] + [0] * (idx - 5000)
    elif idx == 5000:
        pass
    else:
        tmp = [0] * (5000 - idx) + tmp[:-(5000 - idx)]
    return tmp


def nmrformer_res(y, type):
    allpeaks = read_floats_from_txt_files()
    newPeaks = []
    peaks = get_peaks(y)
    x = np.linspace(0, 12, 60000)
    for f1 in peaks:
        if y[int(f1 * 5000)] < 0.001:
            continue
        for f2 in allpeaks:
            if abs(f1 - f2) <= 0.04:
                newPeaks.append(f1)
                break
    if type == '细胞/组织':
        apath = './ai_model/meta_list/all_meta_cell.txt'
        cls = 72
        model_p = './ai_model/models/onedTrans_cell'
    elif type == '血清':
        apath = './ai_model/meta_list/all_meta_serum.txt'
        cls = 67
        model_p = './ai_model/models/onedTrans_serum'
    elif type == '尿液':
        apath = './ai_model/meta_list/all_meta_urine.txt'
        cls = 49
        model_p = './ai_model/models/onedTrans_urine'
    elif type == '羊水':
        apath = './ai_model/meta_list/all_meta_af.txt'
        cls = 47
        model_p = './ai_model/models/onedTrans_af'
    all_meta = get_allmeta(apath)
    om = Output_prob(allMeta=all_meta)
    output = om.test_m(model_p, y, newPeaks, cls)
    in_peaks0 = np.array(output.peak)
    labels0 = np.array(output['指认结果'])
    trace1 = go.Scatter(x=x[::-1], y=y[::-1], mode='lines', name='Line', line=dict(color='black'))
    trace2 = go.Scatter(x=in_peaks0[::-1], y=[y[int(in_peaks0[i] * 5000)] for i in range(len(in_peaks0))][::-1],
                        mode='markers', name='Markers', text=labels0[::-1],
                        marker=dict(size=5))
    # 绘制图表
    data = [trace1, trace2]
    layout = go.Layout(xaxis=dict(
        autorange='reversed'  # 将 x 轴倒序
    ), title='result')
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white',
    )
    plotly_html = fig.to_html(full_html=False)
    return output, plotly_html


# # 计算峰的积分范围的辅助函数
# def find_peak_boundary(xs, spectrum, threshold=0.1):
#     """
#     给定峰的位置，返回该峰的积分范围。
#     threshold 是相对于峰最大值的阈值，通常设为峰最大值的10%。
#     """
#     left = int(xs[0] * 5000 + 5000)  # 获取峰最大值位置的索引
#     right = int(xs[1] * 5000 + 5000)
#     left_peak_height = spectrum[left]
#     right_peak_height = spectrum[right]
#
#     # 计算信号衰减到阈值位置
#     while left > 0 and spectrum[left] > left_peak_height * threshold:
#         left -= 1
#
#     while right < len(spectrum) - 1 and spectrum[right] > right_peak_height * threshold:
#         right += 1
#
#     return left, right
def find_peak_boundary(xs, spectrum, d_threshold_factor=0.001, consecutive_points=3):
    """
    给定峰的参考位置，返回该峰积分区域的左右边界索引。

    参数：
        xs: 长度为2的序列，表示峰左右侧的参考位置（例如经过拟合得到的参数），
            通过公式 int(x * 5000 + 5000) 得到对应的谱图索引。
        spectrum: 一维谱图数据（列表或 ndarray）。
        d_threshold_factor: 导数阈值因子，用于计算导数阈值，
            阈值 = (峰高度 - 基线) * d_threshold_factor，默认值 0.01。
        consecutive_points: 要求连续低于阈值的点数，默认值 3。

    返回：
        left, right: 分别为积分区域的左、右边界索引。
    """
    # 根据 xs 计算参考索引（这里与原代码保持一致）
    left_idx = int(xs[0] * 5000 + 5000)
    right_idx = int(xs[1] * 5000 + 5000)

    spectrum = np.array(spectrum)

    # 计算谱图的基线（简单取最小值）
    baseline = np.min(spectrum)

    # 取参考位置处的信号值作为峰的局部高度
    left_peak_value = spectrum[left_idx]
    right_peak_value = spectrum[right_idx]

    # 根据峰高度和基线确定左右两侧的导数阈值
    left_d_threshold = (left_peak_value - baseline) * d_threshold_factor
    right_d_threshold = (right_peak_value - baseline) * d_threshold_factor

    # 计算谱图的梯度（近似导数）
    grad = np.gradient(spectrum)

    # ---------------------------
    # 寻找左侧边界：
    # 从 left_idx 向左扫描，直到连续 consecutive_points 个点的导数绝对值均低于 left_d_threshold
    left = left_idx
    count = 0
    while left > 0:
        if abs(grad[left]) < left_d_threshold:
            count += 1
            if count >= consecutive_points:
                break
        else:
            count = 0
        left -= 1
    left = max(0, left)

    # ---------------------------
    # 寻找右侧边界：
    # 从 right_idx 向右扫描，直到连续 consecutive_points 个点的导数绝对值均低于 right_d_threshold
    right = right_idx
    count = 0
    while right < len(spectrum) - 1:
        if abs(grad[right]) < right_d_threshold:
            count += 1
            if count >= consecutive_points:
                break
        else:
            count = 0
        right += 1
    right = min(len(spectrum) - 1, right)

    return left, right


def merge_peaks(peaks, metabolites, threshold):
    peaks.insert(0, 0)
    metabolites.insert(0, 'TSP')
    merged_meta = []
    merged_metabolite_ranges = []

    current_metabolite = metabolites[0]
    current_peak_group = [peaks[0]]

    for i in range(1, len(peaks)):
        if metabolites[i] == current_metabolite and abs(peaks[i] - peaks[i - 1]) < threshold:
            current_peak_group.append(peaks[i])
        else:
            merged_meta.append(current_metabolite)
            merged_metabolite_ranges.append((min(current_peak_group), max(current_peak_group)))

            current_metabolite = metabolites[i]
            current_peak_group = [peaks[i]]

    # Add the last group of peaks and metabolite range
    merged_meta.append(current_metabolite)
    merged_metabolite_ranges.append((min(current_peak_group), max(current_peak_group)))

    return merged_meta, merged_metabolite_ranges


def find_integral_region(spectrum, peaks, meta):
    merged_meta, merged_metabolite_ranges = merge_peaks(peaks, meta, 0.04)
    integral_region = []
    for peak_position, metabolite in zip(merged_metabolite_ranges, merged_meta):
        left, right = find_peak_boundary(peak_position, spectrum)
        integral_region.append((left, right))
    return merged_meta, integral_region


# 计算单个峰的积分
def compute_peak_integral(left, right, s, u, spectrum):
    left = int(left * 5000 + 5000) - s
    right = int(right * 5000 + 5000) - s
    x_values = np.linspace(-1, 12, len(spectrum))
    y = np.array(spectrum[left:right + 1]) - u
    integral = np.trapz(y, x=x_values[left:right + 1])
    return integral


def get_spectra(meta):
    spectra = np.zeros(60000)
    peak = []
    folder_path = f'./metabolites_data/{meta}'
    for s in os.listdir(folder_path):
        spc_path = f'./metabolites_data/{meta}/{s}/spectra.txt'
        peak_pth = f'./metabolites_data/{meta}/{s}/peak.txt'
        l = []
        with open(spc_path) as f:
            lines = f.readlines()
            for k in lines:
                l.append(float(k.strip()))
        spectra += np.array(l)
        with open(peak_pth) as f:
            lines = f.readlines()
            for k in lines:
                peak.append(float(k.strip()))
        plotpeak = []
        for i in peak:
            plotpeak.append(spectra[int(i * 5000)])
    return spectra, peak, plotpeak


def plot_sp(meta):
    trace1 = go.Scatter(x=np.linspace(0, 12, 60000), y=get_spectra(meta)[0], mode='lines', name='Line',
                        line=dict(color='black'))
    trace2 = go.Scatter(x=get_spectra(meta)[1], y=get_spectra(meta)[2], mode='markers', name='true',
                        marker=dict(size=6))
    data = [trace1, trace2]
    layout = go.Layout(xaxis=dict(autorange='reversed'))
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white',
    )
    return get_spectra(meta)[0]


def correlation_alignment(arr1, arr2):
    """
    基于相关性的对齐方法
    """
    # 计算互相关函数
    cross_corr = np.correlate(arr1, arr2, mode='full')
    shift = np.argmax(cross_corr) - (len(arr1) - 1)
    # 根据偏移量对arr2进行对齐
    if shift > 0:
        arr2_aligned = np.concatenate((np.zeros(shift), arr2[:-shift]))
    else:
        arr2_aligned = np.concatenate((arr2[-shift:], np.zeros(-shift)))
    return arr2_aligned, shift

def find_smoothest_region(curve, window_size):
    # 计算曲线的导数
    curve_derivative = np.diff(curve)

    # 存储每个窗口的变化率总和
    smoothness_scores = []

    # 滑动窗口计算每个窗口的导数变化率总和
    for i in range(len(curve) - window_size):
        window_derivative = curve_derivative[i:i + window_size - 1]
        smoothness_score = np.sum(np.abs(np.diff(window_derivative)))  # 计算窗口内导数变化的总和
        smoothness_scores.append(smoothness_score)

    # 找到变化率最小的窗口的索引
    min_smoothness_index = np.argmin(smoothness_scores)

    # 返回最平缓区域的中点下标
    return min_smoothness_index + window_size // 2
