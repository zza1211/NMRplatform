# -*- coding: utf-8 -*-
import base64
import csv
import datetime
import io
import json
import os
import shutil
import textwrap
import time
import zipfile

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
import seaborn as sns
import statsmodels.api as sm
from flask import Flask, render_template, request
from flask import jsonify, stream_with_context, Response
from flask import redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_uploads import UploadSet, configure_uploads
from lxml import etree
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from matplotlib_venn import venn2
from openai import OpenAI
from plotly.subplots import make_subplots
from scipy.interpolate import interpolate
from scipy.stats import f_oneway, shapiro, levene, yeojohnson
from scipy.stats import ttest_ind
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from werkzeug.utils import secure_filename

import ai_model.funcs
import nmrutils.brukerprocess
import nmrutils.icoshift
from pypls import cross_validation, plotting

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "metanaly.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads/")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
app = Flask(__name__)
app.secret_key = 'nmr2025'
app.config['UPLOADS_DEFAULT_DEST'] = 'static/upload'
app.config['UPLOADS_DEFAULT_ALLOWED_EXTENSIONS'] = ['csv']
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['PATHWAY_FOLDER'] = 'pathway_file'
app.config['KGML_FOLDER'] = 'pathway_file'
db = SQLAlchemy(app)
files = UploadSet('files')
configure_uploads(app, files)

# 用于存储进度的全局变量
progress = 0
METABOLITE_MAPPING = {}

try:
    with open("meta_keggid_mapping.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['Metabolite'].strip()
            kegg_id = row['KEGG ID'].strip()
            METABOLITE_MAPPING[name] = kegg_id
except Exception as e:
    print("加载 meta_keggid_mapping.csv 时出错:", e)


class FileInfo(db.Model):
    __tablename__ = "file_info"
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(80))
    object = db.Column(db.String(80))
    context = db.Column(db.String(80))
    sample = db.Column(db.String(80))
    gp = db.Column(db.String(80))
    date = db.Column(db.String(80))
    purpose = db.Column(db.String(200))

    def save(self):
        db.session.add(self)
        db.session.commit()
        db.session.close()


# 初始化大模型客户端（请填写您的 api_key 等信息）
client = OpenAI(
    api_key="XXXXXXXXX",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


@app.route('/', endpoint='fileinfo', methods=["GET"])
def hello_world():
    page = request.args.get('page', 1)
    perPage = request.args.get('perPage', 10)
    paginate = FileInfo.query.paginate(page=int(page), per_page=int(perPage))
    return render_template('fileinfo.html', paginate=paginate)


def get_spec(x, y):
    s = -1
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
    x_1 = np.linspace(-1, 12, 65000)
    f = interpolate.interp1d(x, y, kind='cubic', bounds_error=False, fill_value=0)
    y_1 = f(x_1)
    return y_1


@app.route('/upload', methods=['POST'])
def upload():
    global progress

    # # 确保有多个文件上传
    # if 'zip_files[]' not in request.files:
    #     return '没有文件上传'

    # 获取上传的所有文件
    zip_files = request.files.getlist('zip_files[]')
    csv_files = request.files.getlist('csv_files[]')

    upload_dir = os.path.join(UPLOAD_FOLDER, 'tmp')
    os.makedirs(upload_dir, exist_ok=True)
    res = pd.DataFrame()
    # 获取传递的标签
    labels = request.form.getlist('labels[]')
    c = 0
    gpl = []
    if zip_files:
        # 当前文件的索引
        current_file_index = 0
        total_steps = 0
        for zip_file in zip_files:
            # 每个文件的路径
            zip_path = os.path.join(upload_dir, zip_file.filename)
            zip_file.save(zip_path)
            os.makedirs(os.path.join(upload_dir, labels[current_file_index]), exist_ok=True)
            # 解压文件
            extract_zip(zip_path, os.path.join(upload_dir, labels[current_file_index]))
            current_file_index = current_file_index + 1
            os.remove(zip_path)
        # 获取文件夹中的所有数据文件
        datafiles = os.listdir(upload_dir)
        for i, f in enumerate(datafiles):
            if os.path.isdir(os.path.join(upload_dir, f)):
                total_steps = total_steps + len(os.listdir(os.path.join(upload_dir, f)))
                gpl.append(f)
                gpl.append(len(os.listdir(os.path.join(upload_dir, f))))
        for i, f in enumerate(datafiles):
            if os.path.isdir(os.path.join(upload_dir, f)):
                for j, ff in enumerate(os.listdir(os.path.join(upload_dir, f))):
                    sample_data = nmrutils.brukerprocess.data_process(os.path.join(os.path.join(upload_dir, f), ff))
                    res[f"sample_{c + 1}_{f}"] = sample_data  # 为每个样本创建一个列
                    c += 1
                    progress = int(c / total_steps * 100)  # 更新进度

    elif csv_files:
        for csv_file, f in zip(csv_files, labels):
            # Save the CSV file
            csv_path = os.path.join(upload_dir, csv_file.filename)
            csv_file.save(csv_path)

            # Process the CSV content
            df = pd.read_csv(csv_path)
            chemical_shift = df.iloc[:, 0].tolist()  # First column as chemical shifts
            spectra_data = df.iloc[:, 1:].values  # Other columns as spectra for samples
            for i in range(spectra_data.shape[1]):
                c = c + 1
                res[f"sample_{c}_{f}"] = get_spec(spectra_data[:, i], chemical_shift)
                print(res[f"sample_{c}_{f}"])
            gpl.append(f)
            gpl.append(spectra_data.shape[1])
        progress = 1

    #
    shutil.rmtree(upload_dir)  # 删除解压后的文件夹
    #
    #
    # 生成文件名和保存路径
    final_filename = f"research_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    respath = os.path.join(UPLOAD_FOLDER, final_filename)

    # 判断文件名是否重复
    if os.path.exists(respath):
        return "文件名重复"

    # 将结果写入 CSV 文件
    res.to_csv(respath, index=False, encoding='utf_8_sig')

    # 保存文件信息到数据库
    tp = request.form.get('type')
    context = request.form.get('context')
    sample = request.form.get('sample')
    purpose = request.form.get('purpose')
    gp = ''
    for i in range(int(len(gpl) / 2)):
        gp = gp + gpl[2 * i]
        gp = gp + ':'
        gp = gp + str(gpl[2 * i + 1])
        gp = gp + ';'
    fileinfo = FileInfo(
        id=None,
        file_name=final_filename,
        object=sample,
        context=context,
        sample=tp,
        gp=gp,
        date=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        purpose=purpose
    )
    fileinfo.save()

    progress = 100  # 任务完成
    return redirect(url_for('fileinfo'))


@app.route('/delete/<int:id>', methods=['GET'])
def delete_file(id):
    file = FileInfo.query.get(id)
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.file_name))
        tmp_folder = os.path.join(UPLOAD_FOLDER, 'analysis_tmp/', f'{secure_filename(file.file_name).split(".")[0]}/')
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(file_path + '_ali_mod.csv'):
            os.remove(file_path + '_ali_mod.csv')
        if os.path.exists(file_path + '_output.csv'):
            os.remove(file_path + '_output.csv')
        if os.path.exists(file_path + '_output_val.csv'):
            os.remove(file_path + '_output_val.csv')
        if os.path.exists(file_path + '_quantification.csv'):
            os.remove(file_path + '_quantification.csv')
        if os.path.exists(file_path + '_quantregion.csv'):
            os.remove(file_path + '_quantregion.csv')
        if os.path.exists(file_path + '_selected.csv'):
            os.remove(file_path + '_selected.csv')
        if os.path.exists(file_path + 'fin.csv'):
            os.remove(file_path + 'fin.csv')
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
        db.session.delete(file)
        db.session.commit()
        db.session.close()
        flash("文件及其记录已成功删除", "success")
    else:
        flash("文件未找到", "error")
    return redirect(url_for('fileinfo'))


@app.route('/plot/<int:id>', methods=['GET'])
def plot(id):
    file = FileInfo.query.get(id)
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.file_name))
        df = pd.read_csv(file_path)

        # 下采样参数
        original_points = df.shape[0]
        target_points = 5000
        step = original_points // target_points  # 等间隔下采样步长

        # 切片下采样
        df_downsampled = df.iloc[::step, :].reset_index(drop=True)
        x_values = np.linspace(-1, 12, original_points)[::step]

        # 保证x和y长度一致（防止整除不尽）
        if len(x_values) > len(df_downsampled):
            x_values = x_values[:len(df_downsampled)]
        elif len(x_values) < len(df_downsampled):
            df_downsampled = df_downsampled.iloc[:len(x_values), :]

        # 绘图
        fig = go.Figure()
        for column in df_downsampled.columns:
            fig.add_trace(go.Scatter(x=x_values, y=df_downsampled[column], mode='lines', name=column))
        fig.update_xaxes(title_text='ppm', range=[-1, 12])
        fig.update_layout(title=file.file_name, yaxis_title='值')
        fig['layout']['xaxis']['autorange'] = "reversed"
        plotly_html = fig.to_html(full_html=False)
        return render_template('plot.html', plotly_html=plotly_html, file_id=id)
    return redirect(url_for('fileinfo'))


@app.route('/remove_water_peak', methods=['POST'])
def remove_water_peak():
    data = request.get_json()
    peaks = data.get('peaks')
    file_id = int(data.get('file_id'))

    if not peaks or not isinstance(peaks, list):
        return "未提供正确的峰区域数据", 400

    file_info = FileInfo.query.filter_by(id=file_id).first()
    file_path1 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name))
    sps = pd.read_csv(file_path1)

    # 遍历每个峰区域，对每个区域的数据置0
    # 假设谱图行数与 ppm 关系：行索引 j 对应 ppm = (j - 5000) / 5000
    # 因此，给定 ppm 范围 [left, right]，对应行索引范围为：int(left*5000+5000) 到 int(right*5000+5000)
    for peak in peaks:
        left_ppm = float(peak['left_ppm'])
        right_ppm = float(peak['right_ppm'])
        if left_ppm > right_ppm:
            left_ppm, right_ppm = right_ppm, left_ppm
        start_idx = int(left_ppm * 5000 + 5000)
        end_idx = int(right_ppm * 5000 + 5000)
        # 对谱图所有列，在指定行范围内置0
        for col in sps.columns:
            sps.loc[start_idx:end_idx, col] = 0

    # 保存修改后的数据
    sps.to_csv(file_path1, index=False, encoding='utf_8_sig')

    # 返回成功后重定向至绘图页面（可返回JSON格式的重定向地址）
    return jsonify({'redirect_url': url_for('plot', id=file_id)})


@app.route('/analysis/<int:id>', methods=['GET'])
def analysis(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    return render_template('analysis.html', fileinfo=file_info, stat=get_status(file_info.file_name))


@app.route('/nmrformer/<int:id>', methods=['GET'])
def nmrformer(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path1 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name))
    output_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_output_val' + '.csv')
    if not os.path.exists(file_path1):
        flash("文件未找到", "error")
    data = pd.read_csv(file_path1)
    y = data.iloc[:, 0].tolist()
    y = y[5000:]
    m = np.max(y)
    y = [i / m for i in y]
    if os.path.exists(output_path):
        data = pd.read_csv(output_path)[['peak', '指认结果', '概率%', '候选结果', '候选概率%']]
        rows = data.values.tolist()
        headers = ['peak', '指认结果', '概率%', '候选结果', '候选概率%']
        in_peaks0 = np.array(data.peak)
        x = np.linspace(0, 12, 60000)
        labels0 = np.array(data['指认结果'])
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
        return render_template('analysis.html', fileinfo=file_info, ifd='已对齐', plot_html=plotly_html,
                               csv_headers=headers,
                               stat=get_status(file_info.file_name),
                               csv_rows=rows)

    res = ai_model.funcs.nmrformer_res(y, file_info.sample)
    headers = res[0].columns.tolist()
    rows = res[0].values.tolist()
    odata = res[0]
    new_col = [0] * len(odata)
    odata['拟合谱图'] = new_col
    odata.to_csv(output_path, index=False, encoding='utf_8_sig')
    return render_template('analysis.html', fileinfo=file_info, ifd='已对齐', plot_html=res[1], csv_headers=headers,
                           stat=get_status(file_info.file_name),
                           csv_rows=rows)


@app.route('/validate_metabolite', methods=['POST'])
def validate_metabolite():
    data = request.json
    metabolite = data['metabolite']
    peak = float(data['peak'])
    id = int(data['file_id'])
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path1 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name))
    if not os.path.exists(file_path1):
        flash("文件未找到", "error")
    spdata = pd.read_csv(file_path1)
    spe = spdata.iloc[:, 0].tolist()
    spe = spe[5000:]
    ox = np.linspace(peak - 0.05, peak + 0.05, 500)
    oy = spe[int(peak * 5000) - 250:int(peak * 5000) + 250]
    oy = [x / spe[int(peak * 5000)] for x in oy]
    omy, peaks, _ = ai_model.funcs.get_spectra(metabolite)
    fpc = 13
    fp = peak
    for i in range(len(peaks)):
        if abs(peaks[i] - peak) < fpc:
            fp = peaks[i]
            fpc = peaks[i] - peak
    my = omy[int(peak * 5000) - 250:int(peak * 5000) + 250]
    my = [x / omy[int(fp * 5000)] for x in my]

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    axes[0].axvline(x=peak, color='red', linestyle='--', linewidth=1.2, label="Peak Position")
    axes[0].plot(ox, oy, color='black', linewidth=1.2, alpha=0.8)
    # 设置图形属性
    axes[0].set_title(f"Peak Plot for {metabolite}", fontsize=14)
    axes[0].set_xlabel("Chemical Shift (ppm)", fontsize=12)
    axes[0].set_ylabel("Normalized Intensity", fontsize=12)
    axes[0].grid(alpha=0.3)
    omy, peaks, _ = ai_model.funcs.get_spectra(metabolite)
    fpc = 13
    fp = peak
    for i in range(len(peaks)):
        if abs(peaks[i] - peak) < fpc:
            fp = peaks[i]
            fpc = peaks[i] - peak
    my = np.array(omy[int(peak * 5000) - 250:int(peak * 5000) + 250])
    my = my / omy[int(fp * 5000)]
    axes[1].plot(ox, my, color='blue', linewidth=1.2, alpha=0.8)
    axes[1].set_title("参考谱图", fontsize=14)
    axes[1].set_xlabel("Chemical Shift (ppm)", fontsize=12)
    axes[1].set_ylabel("Value", fontsize=12)
    axes[1].grid(alpha=0.3)

    # 调整布局并转换为 PNG
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    png_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    # 返回 Base64 图片给前端
    return jsonify({"image": f"data:image/png;base64,{png_base64}"})


@app.route('/to_adjust/<int:id>', methods=['GET'])
def to_adjust(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path1 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name))
    if not os.path.exists(file_path1):
        flash("文件未找到", "error")
    spdata = pd.read_csv(file_path1)
    spe = spdata.iloc[:, 0].tolist()
    spe = spe[5000:]
    metabolite = request.args.get('metabolite')
    peak = float(request.args.get('peak'))
    ox = np.linspace(peak - 0.05, peak + 0.05, 500)
    oy = spe[int(peak * 5000) - 250:int(peak * 5000) + 250]
    oy = [x / spe[int(peak * 5000)] for x in oy]
    omy, peaks, _ = ai_model.funcs.get_spectra(metabolite)
    fpc = 13
    fp = peak
    for i in range(len(peaks)):
        if abs(peaks[i] - peak) < fpc:
            fp = peaks[i]
            fpc = peaks[i] - peak
    my = omy[int(peak * 5000) - 250:int(peak * 5000) + 250]
    my = [x / omy[int(fp * 5000)] for x in my]
    rd = {"mixed": oy, "pure": my, "x": ox.tolist(), 'peak': peak}
    return render_template('adjust.html', data=rd, id=id)


@app.route('/apply_adjustment', methods=['POST'])
def apply_adjustment():
    data = request.get_json()
    adjusted_spectrum = data.get('adjustedSpectrum')
    id = data.get('id')
    peak = data.get('peak')
    peak_location = data.get('peakLocation')
    file_info = FileInfo.query.filter_by(id=id).first()
    if adjusted_spectrum is None:
        return jsonify({'message': '未接收到谱图数据！'}), 400
    # 处理接收到的谱图（如保存到数据库或进行进一步分析）
    # 保存的逻辑省略
    output_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_ali_mod' + '.csv')
    odata = pd.read_csv(output_path)
    new_col = odata['mod'].tolist()
    new_col1 = odata['not_peak'].tolist()
    peaks = odata['peak'].tolist()
    idx = peaks.index(peak)
    new_col[idx] = adjusted_spectrum
    new_col1[idx] = peak_location
    odata['mod'] = new_col
    odata['not_peak'] = new_col1
    odata.to_csv(output_path, index=False, encoding='utf_8_sig')
    return jsonify({'message': '调整后的谱图已成功保存！'})


@app.route('/update_csv/<int:id>', methods=['POST'])
def update_csv(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    output_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_output' + '.csv')
    output_path1 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_output_val' + '.csv')
    data = request.json.get('data', [])
    if not data or len(data) < 1:
        return jsonify({"error": "无效数据"}), 400

    # 第一行是表头（如果没有表头直接跳过）
    rows = data

    # 初始化保存的结果
    peaks = []
    results = []

    # 处理每行数据（假设第一列是peak，第二列是默认指认结果，最后一列是新增内容）
    for row in rows:
        if len(row) < 3:  # 检查行数据完整性
            return jsonify({"error": "数据格式错误，列数不足"}), 400

        peak = row[0]  # 第一列是 peak
        default_result = row[1]  # 第二列是默认的指认结果
        added_result = row[-2]  # 最后一列是新增内容

        # 优先取“新增内容”列数据，否则取“指认结果”列
        result = added_result.strip() if added_result.strip() else default_result.strip()
        peaks.append(peak)
        results.append(result)

    # 构建保存的 DataFrame
    output_df = pd.DataFrame({
        'peak': peaks,
        '指认结果': results
    })
    odata = pd.read_csv(output_path1)
    odata = odata[['peak', '概率%', '候选结果', '候选概率%', '拟合谱图']]
    output_df['peak'] = output_df['peak'].apply(lambda x: str(x))
    odata['peak'] = odata['peak'].apply(lambda x: str(x))
    odata = pd.merge(output_df, odata, on='peak')
    # 保存到 CSV 文件
    output_df.to_csv(output_path, index=False, encoding='utf_8_sig')
    odata.to_csv(output_path1, index=False, encoding='utf_8_sig')

    return redirect(url_for('analysis', id=id))


@app.route('/meta_select/<int:id>')
def meta_select(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path2 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_output' + '.csv')
    data = pd.read_csv(file_path2)
    data = merge_metabolites(data)
    grouped_data = data.groupby('指认结果').apply(lambda x: x.to_dict(orient='records')).to_dict()
    return render_template('analysis.html', fileinfo=file_info, grouped_data=grouped_data,
                           stat=get_status(file_info.file_name))


# 合并代谢物逻辑
def merge_metabolites(df):
    df = df.sort_values(by=['指认结果', 'peak']).reset_index(drop=True)
    to_remove = set()
    for i in range(len(df) - 1):
        if df.loc[i, '指认结果'] == df.loc[i + 1, '指认结果'] and abs(df.loc[i, 'peak'] - df.loc[i + 1, 'peak']) < 0.04:
            to_remove.add(i + 1)  # 删除后者
    return df.drop(index=to_remove).reset_index(drop=True)


@app.route('/get_plot', methods=['POST'])
def get_plot():
    data = request.json
    metabolite = data['metabolite']
    peak = float(data['peak'])
    id = int(data['file_id'])

    # 获取文件路径并读取数据
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path1 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name))
    sp = pd.read_csv(file_path1)

    # 准备绘图数据
    x_value = np.linspace(peak - 0.05, peak + 0.05, 500)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    colormap = cm.get_cmap('tab10')  # 使用 10 种颜色循环
    p = np.array(sp[sp.columns[0]].tolist()[5000:])[int(peak * 5000)]
    for i, column in enumerate(sp.columns):
        y = np.array(sp[column].tolist()[5000:])
        y_values = y[int(peak * 5000) - 250:int(peak * 5000) + 250] / p
        color = colormap(i % 10)  # 循环分配颜色
        axes[0].plot(x_value, y_values, color=color, linewidth=0.8, alpha=0.6)
    axes[0].axvline(x=peak, color='red', linestyle='--', linewidth=1.2, label="Peak Position")
    # 设置图形属性
    axes[0].set_title(f"Peak Plot for {metabolite}", fontsize=14)
    axes[0].set_xlabel("Chemical Shift (ppm)", fontsize=12)
    axes[0].set_ylabel("Normalized Intensity", fontsize=12)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=10, title="Legend", title_fontsize=12)
    omy, peaks, _ = ai_model.funcs.get_spectra(metabolite)
    fpc = 13
    fp = peak
    for i in range(len(peaks)):
        if abs(peaks[i] - peak) < fpc:
            fp = peaks[i]
            fpc = peaks[i] - peak
    my = np.array(omy[int(peak * 5000) - 250:int(peak * 5000) + 250])
    my = my / omy[int(fp * 5000)]
    axes[1].plot(x_value, my, color='blue', linewidth=1.2, alpha=0.8)
    axes[1].set_title("参考谱图", fontsize=14)
    axes[1].set_xlabel("Chemical Shift (ppm)", fontsize=12)
    axes[1].set_ylabel("Value", fontsize=12)
    axes[1].grid(alpha=0.3)

    # 调整布局并转换为 PNG
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    png_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    # 返回 Base64 图片给前端
    return jsonify({"image": f"data:image/png;base64,{png_base64}"})


@app.route('/save_selection', methods=['POST'])
def save_selection():
    # 获取前端发送的数据
    data = request.json
    selections = data.get('selections', [])  # [{"metabolite": "X", "peak": 0.123}, ...]
    file_id = data.get('file_id')  # 获取 fileinfo.id
    file_info = FileInfo.query.filter_by(id=file_id).first()
    select_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_selected' + '.csv')
    if not selections or not file_id:
        return jsonify({"message": "Invalid request. Missing data."}), 400

    # 创建 DataFrame 并保存为 CSV
    output_df = pd.DataFrame(selections)
    output_df.to_csv(select_path, index=False, encoding='utf_8_sig')

    return jsonify({"message": "Selections saved successfully"})


@app.route('/alig_or_mod/<int:id>')
def alig_or_mod(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name))
    file_path2 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_selected' + '.csv')
    file_path_ali = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_ali_mod' + '.csv')
    meta = pd.read_csv(file_path2)
    if not os.path.exists(file_path_ali):
        data_ali = meta.copy()
        ali = [0] * len(data_ali)
        mod = [0] * len(data_ali)
        upper = [0] * len(data_ali)
        not_peak = [0] * len(data_ali)
        data_ali['shifts'] = ali
        data_ali['mod'] = mod
        data_ali['upper'] = upper
        data_ali['not_peak'] = not_peak
        data_ali.to_csv(file_path_ali, index=False, encoding='utf_8_sig')
    data = pd.read_csv(file_path)
    x = np.linspace(0, 12, 60000)
    y = data.iloc[:, 0].tolist()[5000:]
    in_peaks0 = meta['peak']
    labels0 = meta['metabolite']
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
    headers = meta.columns.tolist()
    rows = meta.values.tolist()
    return render_template('analysis.html', fileinfo=file_info, ali_headers=headers, ali_rows=rows,
                           plotly_html_ali=plotly_html, stat=get_status(file_info.file_name))


@app.route('/ali_peak', methods=['POST'])
def ali_peak():
    data = request.json
    metabolite = data['metabolite']
    peak = float(data['peak'])
    id = int(data['file_id'])

    # 获取文件路径并读取数据
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path1 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name))
    file_path_ali = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_ali_mod' + '.csv')
    sp = pd.read_csv(file_path1)
    meta = pd.read_csv(file_path_ali)
    # 准备绘图数据
    x_value = np.linspace(peak - 0.05, peak + 0.05, 500)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
    colormap = cm.get_cmap('tab10')  # 使用 10 种颜色循环
    p = np.array(sp[sp.columns[0]].tolist()[5000:])[int(peak * 5000)]
    sps = []
    for i, column in enumerate(sp.columns):
        y = np.array(sp[column].tolist()[5000:])
        y_values = y[int(peak * 5000) - 250:int(peak * 5000) + 250] / p
        sps.append(y_values)
        color = colormap(i % 10)  # 循环分配颜色
        axes[0].plot(x_value, y_values, color=color, linewidth=0.8, alpha=0.6)
    axes[0].axvline(x=peak, color='red', linestyle='--', linewidth=1.2, label="Peak Position")
    # 设置图形属性
    axes[0].set_title(f"{metabolite} 对齐前谱图", fontsize=14)
    axes[0].set_xlabel("Chemical Shift (ppm)", fontsize=12)
    axes[0].set_ylabel("Intensity", fontsize=12)
    axes[0].grid(alpha=0.3)
    shifts = [0]
    uppers = [0]
    after_ali = [sps[0]]
    arr1 = sps[0]
    mid_index = ai_model.funcs.find_smoothest_region(arr1, 10)
    for i in range(1, len(sps)):
        arr2 = sps[i]
        # 进行对齐操作
        res = ai_model.funcs.correlation_alignment(arr1, arr2)
        afali = res[0]
        upp = afali[mid_index] - arr1[mid_index]
        uppers.append(upp)
        after_ali.append(afali - upp)
        shifts.append(res[1])
    peaks = meta['peak'].tolist()
    shifts_l = meta['shifts'].tolist()
    uppers_l = meta['upper'].tolist()
    idx = peaks.index(peak)
    shifts_l[idx] = shifts
    uppers_l[idx] = uppers
    meta['shifts'] = shifts_l
    meta['upper'] = uppers_l
    meta.to_csv(file_path_ali, index=False, encoding='utf_8_sig')
    for i in range(len(after_ali)):
        y_values = after_ali[i]
        color = colormap(i % 10)  # 循环分配颜色
        axes[1].plot(x_value, y_values, color=color, linewidth=0.8, alpha=0.6)
    axes[1].set_title("对齐后谱峰", fontsize=14)
    axes[1].set_xlabel("Chemical Shift (ppm)", fontsize=12)
    axes[1].set_ylabel("Intensity", fontsize=12)
    axes[1].grid(alpha=0.3)

    # 调整布局并转换为 PNG
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    png_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    # 返回 Base64 图片给前端
    return jsonify({"image": f"data:image/png;base64,{png_base64}"})


@app.route('/get_region/<int:id>')
def get_region(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path1 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name))
    file_path2 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_ali_mod' + '.csv')
    file_path3 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_output' + '.csv')

    x_value = np.linspace(-1, 12, 65000)
    data = pd.read_csv(file_path1)

    # 处理y_values，绘制每一列数据
    fig = go.Figure()
    y_values = data.iloc[:, 0].tolist()
    fig.add_trace(go.Scatter(x=x_value, y=y_values, mode='lines'))

    if not os.path.exists(file_path2):
        flash('请先选择积分谱峰', 'error')
    op = pd.read_csv(file_path2)
    op['mod'] = op['mod'].astype(str)
    peaks = op[op['mod'] == '0'].peak.tolist()
    meta = op[op['mod'] == '0']['metabolite'].tolist()
    npeaks = []
    nmeta = []
    data_w = pd.read_csv(file_path3)
    for i in range(len(data_w)):
        if data_w['指认结果'][i] in meta and abs(data_w['peak'][i] - peaks[meta.index(data_w['指认结果'][i])]) < 0.04:
            npeaks.append(data_w['peak'][i])
            nmeta.append(data_w['指认结果'][i])
    merged_meta, integral_region = ai_model.funcs.find_integral_region(y_values, npeaks, nmeta)

    # 绘制积分区域两侧的离散点
    for idx, (region, name) in enumerate(zip(integral_region, merged_meta)):
        # 为每个区域分配不同的颜色
        point_color = f"rgba({(idx * 50) % 255}, {(idx * 100) % 255}, {(idx * 150) % 255}, 0.7)"

        # 在积分区域的两侧添加点
        left_x = x_value[region[0]]  # 积分区域左侧的x值
        right_x = x_value[region[1]]  # 积分区域右侧的x值

        # 取该区域两侧的y值
        left_y = y_values[region[0]]
        right_y = y_values[region[1]]

        # 绘制左侧点
        fig.add_trace(go.Scatter(
            x=[left_x],  # 左侧x值
            y=[left_y],  # 左侧y值
            mode='markers',  # 离散点
            marker=dict(color=point_color, size=10),  # 点的颜色和大小
            name=f'{name} 左侧'
        ))

        # 绘制右侧点
        fig.add_trace(go.Scatter(
            x=[right_x],  # 右侧x值
            y=[right_y],  # 右侧y值
            mode='markers',  # 离散点
            marker=dict(color=point_color, size=10),  # 点的颜色和大小
            name=f'{name} 右侧'
        ))

    plot_html = fig.to_html(full_html=False)

    ppm_region = [(x_value[r[0]], x_value[r[1]]) for r in integral_region]
    integral_regions = list(zip(merged_meta, ppm_region))
    return render_template('analysis.html', fileinfo=file_info, plot_html1=plot_html, integral_regions=integral_regions,
                           stat=get_status(file_info.file_name), ifd='已对齐')


@app.route('/save_regions/<int:id>', methods=['POST'])
def save_regions(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    region_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_quantregion' + '.csv')

    data = request.json.get('data', [])
    formatted_data = [{"metabolite": row[0], "左侧值": row[1], "右侧值": row[2]} for row in data]

    pd.DataFrame(formatted_data).to_csv(region_path, index=False, encoding='utf_8_sig')
    return redirect(url_for('analysis', id=id))


@app.route('/quantification/<int:id>')
def quantification(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path1 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name))
    file_path4 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_quantregion' + '.csv')
    file_path5 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_quantification' + '.csv')
    file_path_ali = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_ali_mod' + '.csv')
    if not os.path.exists(file_path4):
        flash('请先进行区域划分', 'error')
    data = pd.read_csv(file_path1)
    reg = pd.read_csv(file_path4)
    regcol = reg.columns.tolist()
    data_ali_mod = pd.read_csv(file_path_ali)
    reg = pd.merge(data_ali_mod, reg, how='outer', left_on='metabolite', right_on='metabolite')
    reg['shifts'] = reg['shifts'].astype(str)
    reg['mod'] = reg['mod'].astype(str)
    reg.fillna(100, inplace=True)
    left = reg['左侧值'].tolist()
    right = reg['右侧值'].tolist()
    col = data.columns.tolist()

    for i in range(len(col)):
        res = []
        c = 0
        sp = data.iloc[:, i].tolist()
        for k in range(len(sp)):
            if sp[k] < 0:
                sp[k] = 0
        for l, r in zip(left, right):
            if (reg['shifts'][c] != '0' and reg['shifts'][c] != '0.0') and (
                    reg['shifts'][c] != 100 and reg['shifts'][c] != 'nan'):
                s = int(eval(reg['shifts'][c])[i])
                u = int(eval(reg['upper'][c])[i])
            else:
                s = 0
                u = 0
            if (reg['mod'][c] == '0' or reg['mod'][c] == '0.0') or (reg['mod'][c] == 100 or reg['mod'][c] == 'nan'):
                res.append(abs(ai_model.funcs.compute_peak_integral(l, r, s, u, sp)))
                c = c + 1
            else:
                not_p = float(reg['not_peak'][c])
                bs = np.array(eval(reg['mod'][c]))
                bs = np.roll(bs, -s)

                p = int(reg['peak'][c] * 5000 + 5000) - s
                not_p = int(not_p * 5000 + 5000) - s
                bs = (bs / bs[249 + not_p - p]) * (sp[not_p] - u)
                x_values = np.linspace(-1, 12, 65000)
                integral = np.trapz(bs, x=x_values[p - 250:p + 250])
                res.append(abs(integral))
                c = c + 1
        reg[col[i]] = res
    reg[regcol + col].to_csv(file_path5, index=False, encoding='utf_8_sig')
    fin = reg[col]
    first_row = fin.iloc[0]
    fin = fin.div(first_row, axis=1)
    fin = fin[1:]
    fin = fin.round(4)
    fin.insert(0, 'metabolite', reg['metabolite'][1:])
    headers = fin.columns.tolist()
    rows = fin.values.tolist()
    return render_template('analysis.html', fileinfo=file_info,
                           stat=get_status(file_info.file_name), ifd='已对齐', qheaders=headers, qrows=rows)


@app.route('/findata/<int:id>')
def findata(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    # 构造归一化后文件的路径（分别为TSP和全谱归一化结果）
    file_fin = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + 'fin' + '.csv')
    normalize_status = None

    if os.path.exists(file_fin):
        data_n = pd.read_csv(file_fin)
        normalize_status = "已进行归一化"
    else:
        # 若未归一化，则读取原始定量文件，展示原始数据
        data_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_quantification' + '.csv')
        data = pd.read_csv(data_path)
        data_value = data[data.columns[3:]].values.T
        data_value = np.around(data_value, 2)
        data_n = pd.DataFrame(data_value, columns=data['metabolite'].tolist())
        group = []
        for col in data.columns[3:]:
            x = col[7:]
            group.append(x[x.index('_') + 1:])
        data_n = data_n.round(4)
        data_n.insert(loc=0, column='组别', value=group)
        data_n.insert(loc=0, column='序号', value=range(1, len(data_n) + 1))
    headers = data_n.columns.tolist()
    rows = data_n.values.tolist()
    return render_template('metanaly.html', headers=headers, rows=rows, id=id, normalize_status=normalize_status)


@app.route('/normalize/<int:id>/<string:method>', methods=['GET', 'POST'])
def normalize(id, method):
    file_info = FileInfo.query.filter_by(id=id).first()
    file_fin = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + 'fin' + '.csv')

    if method == 'tsp':
        # 如果为GET请求，先渲染上传重量文件的页面
        if request.method == 'POST':

            # POST请求：获取上传的重量文件
            weight_file = request.files.get('weight_file')
            if not weight_file:
                return "未上传重量文件", 400

            # 读取定量数据
            data_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_quantification' + '.csv')
            data = pd.read_csv(data_path)
            data_value = data[data.columns[3:]].values.T
            data_value = np.around(data_value, 2)
            df = pd.DataFrame(data_value, columns=data['metabolite'].tolist())

            # 生成分组信息（这里保留原有逻辑）
            group = []
            for col in data.columns[3:]:
                x = col[7:]
                group.append(x[x.index('_') + 1:])

            # 读取上传的重量文件
            try:
                weight_df = pd.read_csv(weight_file)
            except Exception as e:
                return f"读取重量文件失败：{e}", 400

            # 检查重量文件是否只有一列
            if weight_df.shape[1] != 1:
                return "重量文件应只包含一列", 400
            # 检查样本数量是否匹配
            if len(weight_df) != len(df):
                return "样本数量与重量文件数量不匹配", 400

            # 提取重量数据
            weights = weight_df.iloc[:, 0].values

            # 按TSP归一化：先每行除以对应的TSP峰面积，再除以样本对应的重量
            norm_df = df.div(df['TSP'], axis=0)
            norm_df = norm_df.div(weights, axis=0)
            normalize_status = "已进行 TSP归一化（同时除以重量）"
        else:
            # 读取定量数据
            data_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_quantification' + '.csv')
            data = pd.read_csv(data_path)
            data_value = data[data.columns[3:]].values.T
            data_value = np.around(data_value, 2)
            df = pd.DataFrame(data_value, columns=data['metabolite'].tolist())

            # 生成分组信息（这里保留原有逻辑）
            group = []
            for col in data.columns[3:]:
                x = col[7:]
                group.append(x[x.index('_') + 1:])

            # 按TSP归一化：先每行除以对应的TSP峰面积，再除以样本对应的重量
            norm_df = df.div(df['TSP'], axis=0)
            normalize_status = "已进行 TSP归一化"

    elif method == 'full':
        # 使用完整核磁谱图数据进行全谱归一化（原有代码不变）
        file_path1 = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name))
        full_spec = pd.read_csv(file_path1)
        x = np.linspace(-1, 12, 65000)
        integrals = full_spec.apply(lambda col: np.trapz(col, x), axis=0)

        data_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + '_quantification' + '.csv')
        data = pd.read_csv(data_path)
        data_value = data[data.columns[3:]].values.T
        df = pd.DataFrame(data_value, columns=data['metabolite'].tolist())

        group = []
        for col in data.columns[3:]:
            x = col[7:]
            group.append(x[x.index('_') + 1:])

        if len(integrals) != len(df):
            return "样本数量不匹配", 400

        norm_df = df.div(integrals.values, axis=0)
        norm_df = norm_df.apply(lambda col: col * 100)
        normalize_status = "已进行 全谱归一化"
    else:
        return "无效的归一化方法", 400

    # 删除TSP列，并插入分组与序号
    norm_df = norm_df.drop('TSP', axis=1)
    norm_df.insert(loc=0, column='组别', value=group)
    norm_df.insert(loc=0, column='序号', value=range(1, len(norm_df) + 1))

    # 保存归一化后的数据
    norm_df.to_csv(file_fin, index=False, encoding='utf_8_sig')
    headers = norm_df.columns.tolist()
    rows = norm_df.values.tolist()
    return render_template('metanaly.html', headers=headers, rows=rows, id=id, normalize_status=normalize_status)


def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


@app.route('/pca_image/<int:id>')
def pca_image(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + 'fin' + '.csv')
    df = pd.read_csv(file_path)

    group_col = "组别"
    features = df.drop(columns=["序号", group_col]).columns
    X = df[features].fillna(0)
    groups = df[group_col]
    X_scaled = StandardScaler().fit_transform(X)

    # PCA 分析
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_

    # 创建 PCA 总览图
    fig = make_subplots(rows=5, cols=5, subplot_titles=[f"PC{i + 1}" for i in range(5)])
    for i in range(5):
        for j in range(5):
            if i == j:
                # 对角线：显示主成分占比标签
                text = f"PC{i + 1}<br>{explained_variance[i] * 100:.2f}%"
                fig.add_trace(
                    go.Scatter(
                        x=[0], y=[0],
                        mode="text",
                        text=[text],
                        showlegend=False
                    ),
                    row=i + 1, col=j + 1
                )
            else:
                # 非对角线：绘制二维散点图
                scatter_fig = px.scatter(
                    x=X_pca[:, i], y=X_pca[:, j], color=groups,
                    labels={'x': f"PC{i + 1}", 'y': f"PC{j + 1}"}
                )
                for trace in scatter_fig.data:
                    fig.add_trace(trace, row=i + 1, col=j + 1)
    fig.update_layout(showlegend=False)
    fig.update_layout(height=1000, width=1000, title="PCA Overview")

    pca_html = fig.to_html(full_html=False)

    # 创建碎石图
    scree_fig = go.Figure()
    scree_fig.add_trace(go.Scatter(
        x=[f"PC{i + 1}" for i in range(5)],
        y=explained_variance * 100,
        mode='lines+markers',
        marker=dict(size=8),
        line=dict(width=2),
        name='Explained Variance'
    ))
    scree_fig.update_layout(
        title="Scree Plot",
        xaxis_title="Principal Components",
        yaxis_title="Explained Variance (%)",
        height=500, width=800
    )

    scree_html = scree_fig.to_html(full_html=False)

    return render_template(
        'pca.html',
        id=id,
        pca_html=pca_html,
        scree_html=scree_html
    )


# 添加椭圆绘制函数
def plot_point_cov(points, nstd=3, ax=None, **kwargs):
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=3, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ellip)
    return ellip


@app.route('/generate_custom_plot', methods=['POST'])
def generate_custom_plot():
    data = request.get_json()
    x_axis = data['xAxis']
    y_axis = data['yAxis']
    confidence_ellipse_flag = data['confidenceEllipse']
    dpi = data['dpi']
    id = data['id']

    file_info = FileInfo.query.filter_by(id=id).first()
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + 'fin' + '.csv')
    df = pd.read_csv(file_path)

    group_col = "组别"
    features = df.drop(columns=["序号", group_col]).columns
    X = df[features].fillna(0)
    groups = df[group_col]
    X_scaled = StandardScaler().fit_transform(X)

    # PCA 分析
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)

    # 绘制图像
    plt.figure(figsize=(8, 6), dpi=dpi)
    ax = plt.gca()
    sns.scatterplot(x=X_pca[:, x_axis], y=X_pca[:, y_axis], hue=groups, palette="Set1", ax=ax)
    plt.xlabel(f"PC{x_axis + 1}")
    plt.ylabel(f"PC{y_axis + 1}")

    # 绘制置信椭圆
    if confidence_ellipse_flag:
        unique_groups = np.unique(groups)
        colors = sns.color_palette("Set1", len(unique_groups))
        for group, color in zip(unique_groups, colors):
            group_data = X_pca[groups == group]
            plot_point_cov(group_data[:, [x_axis, y_axis]], nstd=2, ax=ax, edgecolor=color, facecolor=color, alpha=0.25)

    # 将图像保存为 Base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return {"image": base64_image}


def opls_da_analysis(data_df):
    X = data_df.iloc[:, 2:].values
    y = data_df.iloc[:, 1].values
    # X = StandardScaler().fit_transform(X)
    X = np.ascontiguousarray(X)
    y_encoded = np.ascontiguousarray(y)
    cv = cross_validation.CrossValidation(kfold=5, estimator="opls")
    cv.fit(X, y_encoded)
    cv.permutation_test()
    plots = plotting.Plots(cv)
    p2 = plots.permutation_plot()
    score_buf = io.BytesIO()
    p2.savefig(score_buf, format='png')
    p2.close()
    score_buf.seek(0)
    perm_img = base64.b64encode(score_buf.read()).decode('utf-8')
    score_buf.close()
    p3, vips = plots.vip_plot(xname="coef", feature_names=data_df.columns[2:])
    score_buf = io.BytesIO()
    p3.savefig(score_buf, format='png')
    p3.close()
    score_buf.seek(0)
    vip_img = base64.b64encode(score_buf.read()).decode('utf-8')
    score_buf.close()
    try:
        p1 = plots.plot_scores()
    except:
        cv.reset_optimal_num_component(2)
        plots = plotting.Plots(cv)
        p1 = plots.plot_scores()
    score_buf = io.BytesIO()
    p1.savefig(score_buf, format='png')
    p1.close()
    score_buf.seek(0)
    score_img = base64.b64encode(score_buf.read()).decode('utf-8')
    score_buf.close()
    # 获取组别信息
    group1, group2 = np.unique(y)
    group1_data = data_df[y == group1].iloc[:, 2:]
    group2_data = data_df[y == group2].iloc[:, 2:]

    # 计算组间均值差异
    group1_means = group1_data.mean()
    group2_means = group2_data.mean()
    means_diff = (group1_means - group2_means).values

    # 新增排序逻辑（关键修改部分）
    # 将代谢物名称、VIP值、浓度差异组合成元组列表
    sorted_data = sorted(zip(data_df.columns[2:], vips, means_diff),
                         key=lambda x: x[1])  # 按VIP值升序排序

    # 解包排序后的数据
    sorted_names = [item[0] for item in sorted_data]
    sorted_vips = [item[1] for item in sorted_data]
    sorted_diff = [item[2] for item in sorted_data]

    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 5))

    # 生成y轴位置（根据排序后的数据长度）
    y_pos = np.arange(len(sorted_vips))

    # 绘制散点图（使用排序后的数据）
    scatter = ax.scatter(sorted_vips,
                         y_pos,
                         c=sorted_diff,  # 使用排序后的浓度差异
                         cmap='coolwarm',
                         norm=Normalize(vmin=-1, vmax=1),
                         s=80,
                         edgecolor='w')

    # 设置坐标轴（使用排序后的名称）
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)  # 使用排序后的代谢物名称
    ax.set_xlabel('VIP Value')
    ax.axvline(x=1, color='grey', linestyle='--')

    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(f'Concentration Difference ({group1} - {group2})')

    # 保存新图像
    score_buf = io.BytesIO()
    fig.savefig(score_buf, format='png', bbox_inches='tight')
    plt.close(fig)
    score_buf.seek(0)
    vip_scatter_img = base64.b64encode(score_buf.read()).decode('utf-8')
    score_buf.close()

    return {
        "score_img": score_img,
        "perm_img": perm_img,
        "vip_img": vip_img,
        "vip_scatter_img": vip_scatter_img
    }, vips  # 注意：返回的vips保持原始顺序


@app.route('/oplsda/<int:id>', methods=['GET', 'POST'])
def oplsda(id):
    # 根据文件id获取文件信息及路径
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + 'fin' + '.csv')
    tmp_folder = os.path.join(UPLOAD_FOLDER, 'analysis_tmp/', f'{secure_filename(file_info.file_name).split(".")[0]}/')
    os.makedirs(tmp_folder, exist_ok=True)
    # 读取csv文件，注意csv中必须有“组别”这一列
    df = pd.read_csv(file_path)
    groups = df['组别'].unique().tolist()  # 获取所有分组

    if request.method == 'POST':
        # 获取用户选定的两个类别（注意不能选择相同的类别）
        selected_group1 = request.form.get('group1')
        selected_group2 = request.form.get('group2')
        if selected_group1 == selected_group2:
            error = "请选择两个不同的类别进行分析。"
            return render_template('oplsda.html', groups=groups, analyzed=False, error=error, id=id)
        # 筛选只包含选定两个类别的数据
        df_filtered = df[df['组别'].isin([selected_group1, selected_group2])]
        # 调用pls_da_analysis函数，返回图像及VIP值数据
        result_images, vips = opls_da_analysis(df_filtered)
        feature_names = df.columns[2:]
        vip_table = []
        for i in range(len(vips)):
            if vips[i] > 1:
                vip_table.append((feature_names[i], vips[i]))
        tmp_path = os.path.join(UPLOAD_FOLDER, 'analysis_tmp/',
                                f'{secure_filename(file_info.file_name).split(".")[0]}/',
                                f'oplsda_{selected_group1 + "VS" + selected_group2}.csv')
        pd.DataFrame({'代谢物': feature_names, 'VIP值': vips}).to_csv(tmp_path, index=False, encoding='utf_8_sig')
        return render_template('oplsda.html',
                               groups=groups,
                               analyzed=True,
                               selected_group1=selected_group1,
                               selected_group2=selected_group2,
                               result_images=result_images,
                               vip_table=vip_table,
                               id=id)
    return render_template('oplsda.html', groups=groups, analyzed=False, id=id)


def pls_da_analysis(data_df):
    X = data_df.iloc[:, 2:].values
    y = data_df.iloc[:, 1].values
    # X = StandardScaler().fit_transform(X)
    X = np.ascontiguousarray(X)
    y_encoded = np.ascontiguousarray(y)
    cv = cross_validation.CrossValidation(kfold=5, estimator="pls")
    cv.fit(X, y_encoded)
    cv.permutation_test()
    plots = plotting.Plots(cv)
    p2 = plots.permutation_plot()
    score_buf = io.BytesIO()
    p2.savefig(score_buf, format='png')
    p2.close()
    score_buf.seek(0)
    perm_img = base64.b64encode(score_buf.read()).decode('utf-8')
    score_buf.close()
    p3, vips = plots.vip_plot(xname="coef", feature_names=data_df.columns[2:])
    score_buf = io.BytesIO()
    p3.savefig(score_buf, format='png')
    p3.close()
    score_buf.seek(0)
    vip_img = base64.b64encode(score_buf.read()).decode('utf-8')
    score_buf.close()
    try:
        p1 = plots.plot_scores()
    except:
        cv.reset_optimal_num_component(2)
        plots = plotting.Plots(cv)
        p1 = plots.plot_scores()
    score_buf = io.BytesIO()
    p1.savefig(score_buf, format='png')
    p1.close()
    score_buf.seek(0)
    score_img = base64.b64encode(score_buf.read()).decode('utf-8')
    score_buf.close()
    # 获取组别信息
    group1, group2 = np.unique(y)
    group1_data = data_df[y == group1].iloc[:, 2:]
    group2_data = data_df[y == group2].iloc[:, 2:]

    # 计算组间均值差异
    group1_means = group1_data.mean()
    group2_means = group2_data.mean()
    means_diff = (group1_means - group2_means).values

    # 新增排序逻辑（关键修改部分）
    # 将代谢物名称、VIP值、浓度差异组合成元组列表
    sorted_data = sorted(zip(data_df.columns[2:], vips, means_diff),
                         key=lambda x: x[1])  # 按VIP值升序排序

    # 解包排序后的数据
    sorted_names = [item[0] for item in sorted_data]
    sorted_vips = [item[1] for item in sorted_data]
    sorted_diff = [item[2] for item in sorted_data]

    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 5))

    # 生成y轴位置（根据排序后的数据长度）
    y_pos = np.arange(len(sorted_vips))

    # 绘制散点图（使用排序后的数据）
    scatter = ax.scatter(sorted_vips,
                         y_pos,
                         c=sorted_diff,  # 使用排序后的浓度差异
                         cmap='coolwarm',
                         norm=Normalize(vmin=-1, vmax=1),
                         s=80,
                         edgecolor='w')

    # 设置坐标轴（使用排序后的名称）
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)  # 使用排序后的代谢物名称
    ax.set_xlabel('VIP Value')
    ax.axvline(x=1, color='grey', linestyle='--')

    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(f'Concentration Difference ({group1} - {group2})')

    # 保存新图像
    score_buf = io.BytesIO()
    fig.savefig(score_buf, format='png', bbox_inches='tight')
    plt.close(fig)
    score_buf.seek(0)
    vip_scatter_img = base64.b64encode(score_buf.read()).decode('utf-8')
    score_buf.close()

    return {
        "score_img": score_img,
        "perm_img": perm_img,
        "vip_img": vip_img,
        "vip_scatter_img": vip_scatter_img
    }, vips  # 注意：返回的vips保持原始顺序


@app.route('/plsda/<int:id>', methods=['GET', 'POST'])
def plsda(id):
    # 根据文件id获取文件信息及路径
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + 'fin' + '.csv')
    tmp_folder = os.path.join(UPLOAD_FOLDER, 'analysis_tmp/', f'{secure_filename(file_info.file_name).split(".")[0]}/')
    os.makedirs(tmp_folder, exist_ok=True)
    # 读取csv文件，注意csv中必须有“组别”这一列
    df = pd.read_csv(file_path)
    groups = df['组别'].unique().tolist()  # 获取所有分组

    if request.method == 'POST':
        # 获取用户选定的两个类别（注意不能选择相同的类别）
        selected_group1 = request.form.get('group1')
        selected_group2 = request.form.get('group2')
        if selected_group1 == selected_group2:
            error = "请选择两个不同的类别进行分析。"
            return render_template('plsda.html', groups=groups, analyzed=False, error=error, id=id)
        # 筛选只包含选定两个类别的数据
        df_filtered = df[df['组别'].isin([selected_group1, selected_group2])]
        # 调用pls_da_analysis函数，返回图像及VIP值数据
        result_images, vips = pls_da_analysis(df_filtered)
        feature_names = df.columns[2:]
        vip_table = []
        for i in range(len(vips)):
            if vips[i] > 1:
                vip_table.append((feature_names[i], vips[i]))
        tmp_path = os.path.join(UPLOAD_FOLDER, 'analysis_tmp/',
                                f'{secure_filename(file_info.file_name).split(".")[0]}/',
                                f'plsda_{selected_group1 + "VS" + selected_group2}.csv')
        pd.DataFrame({'代谢物': feature_names, 'VIP值': vips}).to_csv(tmp_path, index=False, encoding='utf_8_sig')
        return render_template('plsda.html',
                               groups=groups,
                               analyzed=True,
                               selected_group1=selected_group1,
                               selected_group2=selected_group2,
                               result_images=result_images,
                               vip_table=vip_table,
                               id=id)
    return render_template('plsda.html', groups=groups, analyzed=False, id=id)


def generate_venn_diagram(set_left, set_right):
    """
    使用matplotlib_venn生成韦恩图，并返回base64编码的图像字符串
    set_left：左侧圆的代谢物集合（组别1与空白对照的VIP>1代谢物）
    set_right：右侧圆的代谢物集合（组别2与空白对照的VIP>1代谢物）
    """
    plt.figure(figsize=(6, 6))
    v = venn2([set_left, set_right], set_labels=('组别1', '组别2'))
    # 分别计算各区域代谢物集合
    only_left = set_left - set_right
    only_right = set_right - set_left
    intersection = set_left & set_right

    # 为防止名称过长，这里使用textwrap进行换行处理
    def wrap_names(names_set):
        if names_set:
            names_str = ", ".join(sorted(names_set))
            return "\n".join(textwrap.wrap(names_str, width=15))
        else:
            return ""

    if v.get_label_by_id('10'):
        v.get_label_by_id('10').set_text(wrap_names(only_left))
    if v.get_label_by_id('01'):
        v.get_label_by_id('01').set_text(wrap_names(only_right))
    if v.get_label_by_id('11'):
        v.get_label_by_id('11').set_text(wrap_names(intersection))
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64


@app.route('/generate_venn/<int:id>', methods=['POST'])
def generate_venn(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + 'fin' + '.csv')
    df = pd.read_csv(file_path)
    feature_names = df.columns[2:]
    blank = request.form.get('blank')
    group1 = request.form.get('group1')
    group2 = request.form.get('group2')

    df_left = df[df['组别'].isin([blank, group1])]
    df_right = df[df['组别'].isin([blank, group2])]
    if 'oplsda' in request.referrer:
        _, vips_left = opls_da_analysis(df_left)
        _, vips_right = opls_da_analysis(df_right)
    else:
        _, vips_left = pls_da_analysis(df_left)
        _, vips_right = pls_da_analysis(df_right)
    metabolites_left = []
    for i in range(len(vips_left)):
        if vips_left[i] > 1:
            metabolites_left.append(feature_names[i])
    metabolites_right = []
    for i in range(len(vips_right)):
        if vips_right[i] > 1:
            metabolites_right.append(feature_names[i])

    venn_img = generate_venn_diagram(set(metabolites_left), set(metabolites_right))
    return jsonify({'venn_img': venn_img})


@app.route('/anova/<int:id>')
def anova(id):
    # 读取文件
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + 'fin' + '.csv')
    tmp_folder = os.path.join(UPLOAD_FOLDER, 'analysis_tmp/', f'{secure_filename(file_info.file_name).split(".")[0]}/')
    os.makedirs(tmp_folder, exist_ok=True)
    df = pd.read_csv(file_path)

    # 使用第二列作为分组变量
    all_groups = list(df.iloc[:, 1].unique())
    all_groups.sort()

    # 连续变量从第三列开始
    continuous_vars = df.columns[2:]

    # 初始化结果表
    anova_table = []  # 保存 F统计量 和 p-value
    effect_size_table = []  # 保存效应量（Eta Squared）
    diagnostic_table = []  # 保存正态性和方差齐性检验结果，用于表格展示

    # 用于小提琴图的原始数据
    t1, t2, t3, t4 = [], [], [], []

    # 对每个连续变量做处理
    for col in continuous_vars:
        corrected = False  # 标记是否对该变量做了校正
        # 针对每个分组提取原始数据
        group_data = {g: df[df.iloc[:, 1] == g][col].dropna() for g in all_groups}

        # 对每个分组进行正态性检验，并保存结果
        for g, data in group_data.items():
            if len(data) >= 3:  # Shapiro 检验通常要求样本数>=3
                stat, p = shapiro(data)
            else:
                p = np.nan
            diagnostic_table.append(
                {'metabolite': col, 'group': g, 'shapiro_p': p, 'levene_p': np.nan, 'corrected': False})

        # 对所有分组数据做 Levene 检验（需至少2组）
        non_empty = [data for data in group_data.values() if len(data) > 0]
        if len(non_empty) >= 2:
            stat_levene, p_levene = levene(*non_empty)
        else:
            p_levene = np.nan

        # 更新每一行该变量的 Levene 检验结果
        for item in diagnostic_table:
            if item['metabolite'] == col:
                item['levene_p'] = p_levene

        # 判断是否需要校正：若任一分组的 Shapiro 检验 p < 0.05 或 Levene 检验 p < 0.05
        need_correction = any(((item['shapiro_p'] is not None and item['shapiro_p'] < 0.05)
                               for item in diagnostic_table if item['metabolite'] == col)) \
                          or (p_levene is not None and p_levene < 0.05)

        # 若需要校正，则对整个变量使用 Yeo-Johnson 变换
        if need_correction:
            # 记录校正标志：更新 diagnostic_table 对应记录
            for item in diagnostic_table:
                if item['metabolite'] == col:
                    item['corrected'] = True
            corrected = True
            # 对该变量进行 Yeo-Johnson 变换（仅对非缺失值）
            all_data = df[col].dropna().values
            transformed_data, lmbda = yeojohnson(all_data)
            # 将校正后的数据存回新列，用于后续分析（注意：仅更新非缺失部分）
            df.loc[df[col].notna(), col] = transformed_data
            # 此处可选择再次进行正态性检验与方差齐性检验，但为了简化流程，此处不再重复

        # 生成诊断图（QQ图）：若校正，则展示校正前后的对比，否则仅展示原始数据QQ图
        # fig_diag = plt.figure(figsize=(6 * len(all_groups), 5))
        # ncol = 2 if corrected else 1
        # for i, g in enumerate(all_groups):
        #     # 原始数据或校正后的数据均存于 df 中
        #     data = df[df.iloc[:, 1] == g][col].dropna()
        #     ax = plt.subplot(1, len(all_groups) * ncol, i * ncol + 1)
        #     probplot(data, dist="norm", plot=ax)
        #     ax.set_title(
        #         f"{col} - {g}\n{'校正后' if corrected else '原始'}\nShapiro p={diagnostic_table[i]['shapiro_p']:.2e}")
        #     if corrected:
        #         # 如果校正，则也展示校正前的QQ图（利用保存的原始数据）
        #         original = group_data[g]
        #         ax2 = plt.subplot(1, len(all_groups) * ncol, i * ncol + 2)
        #         probplot(original, dist="norm", plot=ax2)
        #         ax2.set_title(
        #             f"{col} - {g}\n原始数据\nShapiro p={(shapiro(original)[1] if len(original) >= 3 else np.nan):.2e}")
        # plt.tight_layout()
        # buf_diag = io.BytesIO()
        # plt.savefig(buf_diag, format='png')
        # buf_diag.seek(0)
        # diag_img = base64.b64encode(buf_diag.getvalue()).decode('utf-8')
        # diagnostic_plots.append({'metabolite': col, 'image': diag_img, 'corrected': corrected})
        # plt.close(fig_diag)

        # 对校正后的数据，重新按照分组提取数据
        final_group_data = [df[df.iloc[:, 1] == g][col].dropna() for g in all_groups]
        non_empty_final = [d for d in final_group_data if len(d) > 0]
        if len(non_empty_final) >= 2:
            F_stat, p_value = f_oneway(*non_empty_final)
        else:
            F_stat, p_value = np.nan, np.nan

        # 计算效应量 (Eta Squared)
        all_data_final = df[col].dropna().values
        if len(all_data_final) > 0:
            overall_mean = np.mean(all_data_final)
            SSB = 0  # 组间平方和
            for g in all_groups:
                data_g = df[df.iloc[:, 1] == g][col].dropna().values
                n_i = len(data_g)
                if n_i > 0:
                    group_mean = np.mean(data_g)
                    SSB += n_i * (group_mean - overall_mean) ** 2
            SST = np.sum((all_data_final - overall_mean) ** 2)  # 总平方和
            effect_size = SSB / SST if SST > 0 else np.nan
        else:
            effect_size = np.nan

        anova_table.append({'metabolite': col, 'F_stat': F_stat, 'p_value': p_value})
        effect_size_table.append({'metabolite': col, 'effect_size': effect_size})
        t1.append(col)
        t2.append(F_stat)
        t3.append(p_value)
        t4.append(effect_size)

    # 保存ANOVA结果到 CSV
    tmp_path = os.path.join(UPLOAD_FOLDER, 'analysis_tmp/',
                            f'{secure_filename(file_info.file_name).split(".")[0]}/',
                            'anova.csv')
    pd.DataFrame({'代谢物': t1, 'F_stat': t2, 'p_value': t3, 'effect_size': t4}).to_csv(tmp_path, index=False,
                                                                                        encoding='utf_8_sig')

    # 绘制每个连续变量的总体小提琴图（基于最终数据）
    num_vars = len(continuous_vars)
    fig_violin, axs = plt.subplots(nrows=num_vars, ncols=1, figsize=(8, 4 * num_vars))
    if num_vars == 1:
        axs = [axs]
    for i, col in enumerate(continuous_vars):
        data = [df[df.iloc[:, 1] == g][col].dropna() for g in all_groups]
        axs[i].violinplot(data, showmeans=False, showmedians=True)
        axs[i].set_xticks(range(1, len(all_groups) + 1))
        axs[i].set_xticklabels(all_groups)
        axs[i].set_title(col)
        axs[i].set_ylabel('相对浓度')
    plt.tight_layout()
    buf_violin = io.BytesIO()
    plt.savefig(buf_violin, format='png')
    buf_violin.seek(0)
    violin_img = base64.b64encode(buf_violin.getvalue()).decode('utf-8')
    plt.close(fig_violin)

    return render_template('anova.html',
                           violin_img=violin_img,
                           id=id,
                           anova_table=anova_table,
                           effect_size_table=effect_size_table,
                           diagnostic_table=diagnostic_table)
    # diagnostic_plots=diagnostic_plots


@app.route('/ttest/<int:id>')
def ttest(id):
    # 读取文件
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + 'fin' + '.csv')
    tmp_folder = os.path.join(UPLOAD_FOLDER, 'analysis_tmp/', f'{secure_filename(file_info.file_name).split(".")[0]}/')
    os.makedirs(tmp_folder, exist_ok=True)
    df = pd.read_csv(file_path)

    # 提取所有分组（假设第二列为分组信息）
    groups = list(df.iloc[:, 1].unique())
    groups.sort()

    # 从请求参数中获取用户选择的两个分组
    group1 = request.args.get('group1')
    group2 = request.args.get('group2')

    # --------- 生成火山图及结果表格 ---------
    volcano_img = None
    volcano_table = []  # 保存火山图计算结果的数据
    t1, t2, t3 = [], [], []

    if group1 and group2 and group1 in groups and group2 in groups and group1 != group2:
        # 针对每个连续变量进行 t-test 前的检验和校正
        for col in df.columns[2:]:
            corrected = False
            # 提取原始数据（仅考虑选定的两个分组）
            orig_data1 = df[df.iloc[:, 1] == group1][col].dropna()
            orig_data2 = df[df.iloc[:, 1] == group2][col].dropna()
            # 进行 Shapiro 检验（样本数>=3时）
            p1 = shapiro(orig_data1)[1] if len(orig_data1) >= 3 else np.nan
            p2 = shapiro(orig_data2)[1] if len(orig_data2) >= 3 else np.nan
            # Levene 检验（两组数据）
            if len(orig_data1) > 0 and len(orig_data2) > 0:
                p_levene = levene(orig_data1, orig_data2)[1]
            else:
                p_levene = np.nan

            # 判断是否需要校正：若任一组正态性检验或方差齐性检验 p < 0.05
            need_correction = ((p1 is not None and p1 < 0.05) or (p2 is not None and p2 < 0.05)) \
                              or (p_levene is not None and p_levene < 0.05)

            # 提取校正后数据进行 t-test（若已校正，则用更新后的数据，否则仍用原始数据）
            new_data1 = df[df.iloc[:, 1] == group1][col].dropna()
            new_data2 = df[df.iloc[:, 1] == group2][col].dropna()
            if len(new_data1) > 0 and len(new_data2) > 0:
                t_stat, p_val = ttest_ind(new_data1, new_data2)
            else:
                p_val = np.nan

            # 计算均值和 fold change（以 group1 为基准）
            mean1 = new_data1.mean()
            mean2 = new_data2.mean()
            fc = mean2 / mean1 if mean1 != 0 else np.nan
            volcano_table.append({
                'metabolite': col,
                'fold_change': fc,
                'p_value': p_val
            })
            t1.append(col)
            t2.append(fc)
            t3.append(p_val)

        # 保存 t-test 结果到 CSV
        tmp_path = os.path.join(UPLOAD_FOLDER, 'analysis_tmp/',
                                f'{secure_filename(file_info.file_name).split(".")[0]}/',
                                f'ttest_{group1 + "VS" + group2}.csv')
        pd.DataFrame({'代谢物': t1, 'fold_change': t2, 'p_value': t3}).to_csv(tmp_path, index=False,
                                                                              encoding='utf_8_sig')

        # 计算 log2(FC) 与 -log10(p-value)，并标记重要性（例如 p < 0.05）
        for item in volcano_table:
            if np.isnan(item['fold_change']) or item['fold_change'] <= 0 or np.isnan(item['p_value']):
                item['log2_fc'] = None
                item['neg_log10_p'] = None
                item['important'] = False
            else:
                item['log2_fc'] = np.log2(item['fold_change'])
                item['neg_log10_p'] = -np.log10(item['p_value']) if item['p_value'] > 0 else None
                item['important'] = (item['p_value'] < 0.05)

        # 绘制火山图：横坐标为 log2(FC)，纵坐标为 -log10(p-value)
        fig, ax = plt.subplots(figsize=(8, 6))
        important_plotted = False
        non_important_plotted = False
        for item in volcano_table:
            if item['log2_fc'] is None or item['neg_log10_p'] is None:
                continue
            if item['important']:
                if not important_plotted:
                    ax.scatter(item['log2_fc'], item['neg_log10_p'], color='red', s=50, label='重要性代谢物')
                    important_plotted = True
                else:
                    ax.scatter(item['log2_fc'], item['neg_log10_p'], color='red', s=50)
                ax.text(item['log2_fc'], item['neg_log10_p'], item['metabolite'], fontsize=8, color='red')
            else:
                if not non_important_plotted:
                    ax.scatter(item['log2_fc'], item['neg_log10_p'], color='blue', s=30, label='非重要性代谢物')
                    non_important_plotted = True
                else:
                    ax.scatter(item['log2_fc'], item['neg_log10_p'], color='blue', s=30)
                ax.text(item['log2_fc'], item['neg_log10_p'], item['metabolite'], fontsize=8)
        ax.set_xlabel('Log2(Fold Change)')
        ax.set_ylabel('-Log10(p-value)')
        ax.set_title(f'火山图 Volcano Plot ({group1} vs {group2})')
        ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        volcano_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

    # --------- 生成箱线图（带散点，散点置于最前） ---------
    # 仅比较选择的两个分组
    metabolites = df.columns[2:]
    num_metabolites = len(metabolites)

    fig_box, axs = plt.subplots(nrows=num_metabolites, ncols=1, figsize=(8, 4 * num_metabolites))
    if num_metabolites == 1:
        axs = [axs]

    for i, col in enumerate(metabolites):
        data1 = df[df.iloc[:, 1] == group1][col].dropna() if group1 in groups else pd.Series()
        data2 = df[df.iloc[:, 1] == group2][col].dropna() if group2 in groups else pd.Series()
        data = [data1, data2]
        bp = axs[i].boxplot(data, labels=[group1, group2], patch_artist=True)
        axs[i].set_title(col)
        axs[i].set_ylabel('相对浓度')
        # 在箱线图上叠加散点（添加随机抖动，设置 zorder 确保散点在箱体上方）
        for j, d in enumerate(data):
            y = d.values
            x = np.random.normal(j + 1, 0.04, size=len(y))
            axs[i].scatter(x, y, color='black', alpha=0.7, zorder=10)
    plt.tight_layout()
    buf_box = io.BytesIO()
    plt.savefig(buf_box, format='png')
    buf_box.seek(0)
    boxplot_img = base64.b64encode(buf_box.getvalue()).decode('utf-8')
    plt.close(fig_box)

    return render_template('ttest.html',
                           volcano_img=volcano_img,
                           boxplot_img=boxplot_img,
                           id=id,
                           groups=groups,
                           selected1=group1,
                           selected2=group2,
                           volcano_table=volcano_table)


@app.route('/rforest/<int:id>')
def rforest(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + 'fin' + '.csv')
    df = pd.read_csv(file_path)

    # 假设第二列为分组（分类目标），之后的列为特征（各代谢物浓度）
    y = df.iloc[:, 1]
    X = df.iloc[:, 2:]
    features = list(X.columns)

    # 训练随机森林模型，用于 VIP 图和获取 OOB 评分
    rf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    rf.fit(X, y)

    # 计算各特征的重要性得分
    importances = rf.feature_importances_
    rf_results = []
    for feat, imp in zip(features, importances):
        rf_results.append({'feature': feat, 'importance': imp})
    # 按重要性降序排列
    rf_results = sorted(rf_results, key=lambda x: x['importance'], reverse=True)

    # --- VIP 图 ---
    fig, ax = plt.subplots(figsize=(10, 8))
    feat_names = [item['feature'] for item in rf_results]
    imp_values = [item['importance'] for item in rf_results]
    y_pos = range(len(feat_names))
    ax.barh(y_pos, imp_values, align='center', color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names)
    ax.invert_yaxis()  # 最高重要性置于顶部
    ax.set_xlabel('Importance Score')
    ax.set_title('VIP 图 - 随机森林特征重要性')
    # 在条形上添加数值标注
    for i, v in enumerate(imp_values):
        ax.text(v + 0.005, i, f"{v:.3f}", color='blue', va='center')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    vip_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    # --- 决策树样例 ---
    # 选择随机森林中的第一棵树进行展示
    fig_dt, ax_dt = plt.subplots(figsize=(12, 8))
    tree.plot_tree(rf.estimators_[0],
                   filled=True,
                   feature_names=features,
                   class_names=np.unique(y).astype(str),
                   fontsize=8)
    ax_dt.set_title("决策树样例")
    buf_dt = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf_dt, format='png')
    buf_dt.seek(0)
    dtree_img = base64.b64encode(buf_dt.getvalue()).decode('utf-8')
    plt.close(fig_dt)

    # --- 误差变化图 ---
    # 利用 warm_start 记录不同树数下的 OOB 错误率
    rf_ws = RandomForestClassifier(n_estimators=10, warm_start=True, oob_score=True, random_state=42)
    estimator_range = list(range(10, 101, 10))
    oob_errors = []
    for n in estimator_range:
        rf_ws.n_estimators = n
        rf_ws.fit(X, y)
        oob_error = 1 - rf_ws.oob_score_
        oob_errors.append(oob_error)
    fig_err, ax_err = plt.subplots(figsize=(8, 6))
    ax_err.plot(estimator_range, oob_errors, marker='o', linestyle='-')
    ax_err.set_xlabel('Number of Trees')
    ax_err.set_ylabel('OOB Error Rate')
    ax_err.set_title('误差变化图 - 随机森林 OOB Error vs. Number of Trees')
    buf_err = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf_err, format='png')
    buf_err.seek(0)
    error_plot_img = base64.b64encode(buf_err.getvalue()).decode('utf-8')
    plt.close(fig_err)

    # 获取模型 OOB 评分
    oob_score = rf.oob_score_

    return render_template('rforest.html',
                           id=id,
                           vip_img=vip_img,
                           dtree_img=dtree_img,
                           error_plot_img=error_plot_img,
                           rf_table=rf_results,
                           oob_score=oob_score)


@app.route('/kmeans/<int:id>')
def kmeans(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + 'fin' + '.csv')
    df = pd.read_csv(file_path)

    # 用户选择聚类类型和类别数（通过 GET 参数传入）
    # mode: "sample"（样本聚类，默认）或 "attribute"（属性聚类）
    mode = request.args.get('mode', 'sample')
    try:
        n_clusters = int(request.args.get('n_clusters', 3))
        if n_clusters < 2:
            n_clusters = 2
    except:
        n_clusters = 3

    # 根据聚类类型构造数据矩阵
    if mode == 'sample':
        # 假设 CSV 第1列为样本编号，第2列为分组（非聚类数据），之后的列为特征
        data = df.iloc[:, 2:].values  # shape: (n_samples, n_features)
        # 用于绘制均值图时的横坐标：特征名称
        x_labels = list(df.columns[2:])
    else:  # attribute clustering
        # 对属性聚类，将特征矩阵转置：每个“样本”实际上为原来的一个特征
        data = df.iloc[:, 2:].T.values  # shape: (n_features, n_samples)
        # 横坐标：原始样本编号（第一列），转换为字符串
        x_labels = list(df.iloc[:, 0].astype(str).values)

    # 进行 KMeans 聚类
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans_model.fit_predict(data)

    # 计算轮廓系数
    sil_score = silhouette_score(data, cluster_labels)

    # PCA 降维到2D
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    exp_var = pca.explained_variance_ratio_

    # 绘制 PCA 散点图
    import matplotlib.cm as cm
    fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
    colors = cm.get_cmap("viridis", n_clusters)
    for cluster in range(n_clusters):
        idx = cluster_labels == cluster
        ax_pca.scatter(data_pca[idx, 0], data_pca[idx, 1],
                       label=f"Cluster {cluster + 1}",
                       color=colors(cluster))
    ax_pca.set_xlabel(f"PC1 ({exp_var[0] * 100:.1f}%)")
    ax_pca.set_ylabel(f"PC2 ({exp_var[1] * 100:.1f}%)")
    ax_pca.set_title("KMeans 聚类后 PCA 降维 2D 图")
    ax_pca.legend()
    buf_pca = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf_pca, format='png')
    buf_pca.seek(0)
    pca_img = base64.b64encode(buf_pca.getvalue()).decode('utf-8')
    plt.close(fig_pca)

    # 计算各类别均值向量
    cluster_means = []
    unique_clusters = sorted(np.unique(cluster_labels))
    for cluster in unique_clusters:
        cluster_data = data[cluster_labels == cluster]
        mean_vec = np.mean(cluster_data, axis=0)
        cluster_means.append(mean_vec)

    # 绘制各类别均值的折线图
    fig_line, ax_line = plt.subplots(figsize=(10, 6))
    x = range(len(x_labels))
    for i, mean_vec in enumerate(cluster_means):
        ax_line.plot(x, mean_vec, marker='o', label=f"Cluster {i + 1}")
    ax_line.set_xticks(x)
    ax_line.set_xticklabels(x_labels, rotation=45, ha='right')
    ax_line.set_xlabel("特征" if mode == 'sample' else "样本")
    ax_line.set_ylabel("均值")
    ax_line.set_title("各类别均值折线图")
    ax_line.legend()
    buf_line = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf_line, format='png')
    buf_line.seek(0)
    line_img = base64.b64encode(buf_line.getvalue()).decode('utf-8')
    plt.close(fig_line)

    # 构造各类别均值的表格数据（每个类别一行，列为横坐标标签和值）
    cluster_means_table = []
    for i, mean_vec in enumerate(cluster_means):
        row = {"Cluster": f"Cluster {i + 1}"}
        for label, value in zip(x_labels, mean_vec):
            row[label] = value
        cluster_means_table.append(row)

    return render_template('kmeans.html',
                           id=id,
                           mode=mode,
                           n_clusters=n_clusters,
                           pca_img=pca_img,
                           sil_score=sil_score,
                           line_img=line_img,
                           cluster_means_table=cluster_means_table)


@app.route('/logistic/<int:id>', methods=['GET', 'POST'])
def logistic(id):
    # 读取 CSV 文件
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + 'fin' + '.csv')
    data = pd.read_csv(file_path)
    # 假定第一列为序号、第二列为组别，后续列均为代谢物
    group_col = data.columns[1]  # 组别所在列www
    metabolite_columns = data.columns[2:]
    groups = sorted(data[group_col].unique().tolist())

    if request.method == 'POST':
        # 获取前端提交的选择
        selected_metabolites = request.form.getlist('metabolites')
        group1 = request.form.get('group1')
        group2 = request.form.get('group2')

        # 检查是否至少选择了一个代谢物，且两个组别不相同
        if len(selected_metabolites) == 0 or group1 == group2:
            return render_template('logistic.html', metabolites=metabolite_columns, groups=groups,
                                   error="请至少选择一个代谢物，且两个组别必须不同！", id=id)

        # 筛选出所需的两组数据
        subset = data[data[group_col].isin([group1, group2])].copy()
        # 将两组数据转换为二分类，设定 group1 为 0，group2 为 1
        subset['target'] = subset[group_col].apply(lambda x: 0 if x == group1 else 1)

        # 构造自变量 X 和因变量 y
        X = subset[selected_metabolites]
        y = subset['target']

        # 使用 sklearn 进行逻辑回归（这里不输出系数显著性，仅用于预测概率）
        clf = LogisticRegression(solver='liblinear')
        clf.fit(X, y)
        y_pred_proba = clf.predict_proba(X)[:, 1]

        # 计算 ROC 曲线与 AUC
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # 绘制 ROC 曲线，并保存到内存中
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        roc_img = base64.b64encode(buf.getvalue()).decode('utf-8')

        # 利用 statsmodels 计算各变量（包括截距）的 p 值
        X_sm = sm.add_constant(X)
        logit_model = sm.Logit(y, X_sm)
        result = logit_model.fit(disp=0)
        p_values = result.pvalues.to_dict()

        # 返回结果页面，同时将 analysis_id 返回，用于“返回”按钮
        return render_template('results.html', roc_img=roc_img, auc=roc_auc, p_values=p_values, id=id)

    else:
        # GET 请求，返回表单页面
        return render_template('logistic.html', metabolites=metabolite_columns, groups=groups, id=id)


def load_pathway_data(species):
    """加载通路数据"""
    file_path = os.path.join(app.config['PATHWAY_FOLDER'], f"{species}_pathways.csv")
    if not os.path.exists(file_path):
        return None

    pathway_data = pd.read_csv(file_path)
    pathway_data.columns = ['pathway_id', 'pathway_name', 'cids', 'kgml_path']
    pathway_data['cids'] = pathway_data['cids'].str.split(',')
    return pathway_data


def load_metabolite_mapping():
    """加载代谢物名称到Cid的映射"""
    mapping_file = 'meta_keggid_mapping.csv'
    mapping = pd.read_csv(mapping_file)
    return dict(zip(mapping['Metabolite'], mapping['KEGG ID']))


def load_concentration_data(file_path):
    """加载浓度数据并转换代谢物名称为Cid"""
    data = pd.read_csv(file_path)
    name_to_cid = load_metabolite_mapping()

    # 转换列名为Cid
    cid_columns = []
    for col in data.columns[2:]:
        if col in name_to_cid:
            cid_columns.append(name_to_cid[col])
        else:
            cid_columns.append(None)

    # 过滤掉无法映射的列
    valid_indices = [i for i, cid in enumerate(cid_columns) if cid is not None]
    valid_cids = [cid_columns[i] for i in valid_indices]
    valid_data = data.iloc[:, [0, 1] + [i + 2 for i in valid_indices]]
    valid_data.columns = ['序号', '组别'] + valid_cids

    return valid_data


def global_test_enrichment(pathway_data, concentration_data, group1, group2):
    """执行Global Test富集分析"""
    np.random.seed(42)
    results = []

    # 分组数据
    group1_data = concentration_data[concentration_data['组别'] == group1]
    group2_data = concentration_data[concentration_data['组别'] == group2]

    # 计算每组中每个代谢物的平均浓度
    group1_means = group1_data.mean(numeric_only=True)
    group2_means = group2_data.mean(numeric_only=True)

    # 计算差异
    diff = group1_means - group2_means
    abs_diff = abs(diff)

    for _, pathway in pathway_data.iterrows():
        pathway_cids = pathway['cids']
        common_cids = list(set(pathway_cids) & set(concentration_data.columns[2:]))

        # 匹配状态
        match_status = f"{len(common_cids)}/{len(pathway_cids)}"

        if len(common_cids) == 0:
            continue

        # 提取通路中代谢物的差异值
        pathway_diff = diff[common_cids]
        pathway_abs_diff = abs_diff[common_cids]

        # Global Test统计量: 差异绝对值之和
        test_statistic = pathway_abs_diff.sum()

        # 置换检验计算p值
        observed = test_statistic
        n_permutations = 200
        greater = 0
        # 置换检验修改部分：
        group1_size = len(group1_data)
        group2_size = len(group2_data)
        all_indices = concentration_data.index.tolist()

        for _ in range(n_permutations):
            shuffled_indices = np.random.permutation(all_indices)
            shuffled_group1 = concentration_data.loc[shuffled_indices[:group1_size]]
            shuffled_group2 = concentration_data.loc[shuffled_indices[group2_size:]]

            # 重新计算差异
            shuffled_diff = shuffled_group1.mean(numeric_only=True) - shuffled_group2.mean(numeric_only=True)
            shuffled_pathway_diff = shuffled_diff[common_cids]
            shuffled_test_statistic = abs(shuffled_pathway_diff).sum()

            if shuffled_test_statistic >= observed:
                greater += 1

        p_value = (greater + 1) / (n_permutations + 1)
        log_p = -np.log10(p_value) if p_value > 0 else 0

        results.append({
            'pathway_id': pathway['pathway_id'],
            'pathway_name': pathway['pathway_name'],
            'match_status': match_status,
            'pathway_cids': pathway_cids,
            'p_value': p_value,
            'log_p': log_p,
            # 'holm_p': None,
            'fdr': None,
            'piv': None
        })

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # 多重检验校正
    # _, holm_p, _, _ = multipletests(results_df['p_value'], method='holm')
    _, fdr, _, _ = multipletests(results_df['p_value'], method='fdr_bh')

    # results_df['holm_p'] = holm_p
    results_df['fdr'] = fdr

    control_data = concentration_data[concentration_data['组别'] == group1][concentration_data.columns[2:]]
    treatment_data = concentration_data[concentration_data['组别'] == group2][concentration_data.columns[2:]]
    metabolite_changes = np.log2(treatment_data.mean() / control_data.mean())
    # 计算PIV (Pathway Impact Value)
    results_df['piv'] = results_df.apply(
        lambda row: calculate_piv_f(pathway_data[pathway_data['pathway_id'] == row['pathway_id']].iloc[0],
                                    row['pathway_cids'], metabolite_changes), axis=1)
    results_df['piv'] = results_df['piv'] / results_df['piv'].max()
    return results_df


def calculate_relative_betweenness_centrality(G, pathway_cids):
    """计算Relative-betweeness Centrality"""
    # 计算所有节点的介数中心性
    betweenness = nx.betweenness_centrality(G)

    # 计算通路中节点的介数中心性
    pathway_betweenness = {node: betweenness[node] for node in pathway_cids if node in G.nodes}

    # 归一化
    max_bc = max(pathway_betweenness.values()) if pathway_betweenness else 0
    if max_bc == 0:
        return 0

    normalized_bc = {node: bc / max_bc for node, bc in pathway_betweenness.items()}

    return sum(normalized_bc.values()) / len(normalized_bc) if len(normalized_bc) > 0 else 0


def parse_kgml(file_path):
    """解析本地的 KEGG KGML 文件，提取代谢物连接关系"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            kgml_data = f.read()
        root = etree.fromstring(kgml_data.encode())
    except Exception as e:
        print(f"解析 {file_path} 失败: {e}")
        return set(), []

    compound_nodes = {}
    for node in root.findall(".//entry"):
        try:
            if node.get("type") == "compound":
                cid = node.attrib['name'].split(':')[1]
                node_id = node.get("id")
                compound_nodes[node_id] = cid
        except:
            continue

    edges = set()
    for reaction in root.findall(".//reaction"):
        try:
            inputs = [sub.attrib['id'] for sub in reaction.findall("substrate")]
            outputs = [sub.attrib['id'] for sub in reaction.findall("product")]
            for in_node in inputs:
                for out_node in outputs:
                    in_cid = compound_nodes.get(in_node, None)
                    out_cid = compound_nodes.get(out_node, None)
                    if in_cid and out_cid:
                        edges.add((in_cid, out_cid))
        except:
            continue

    return edges, compound_nodes.values()


def build_adjacency_matrix(metabolites, edges):
    """根据代谢物列表和边列表构建邻接矩阵（有向图）"""
    metabolite_list = list(metabolites)
    idx_map = {cid: i for i, cid in enumerate(metabolite_list)}
    num_metabolites = len(metabolite_list)
    adj_matrix = np.zeros((num_metabolites, num_metabolites), dtype=int)

    for src, dst in edges:
        if src in idx_map and dst in idx_map:
            adj_matrix[idx_map[src], idx_map[dst]] = 1
            adj_matrix[idx_map[dst], idx_map[src]] = 1  # 添加反向边
    return adj_matrix, metabolite_list


def calculate_piv(adjacency_matrix, metabolite_changes):
    """计算通路的 PIV（基于节点度和代谢物变化）"""
    degrees = np.sum(adjacency_matrix, axis=1)
    weighted_contributions = degrees * np.abs(metabolite_changes)
    piv = np.sum(weighted_contributions)
    return piv


def calculate_piv_f(pathway, pathway_cids, achanges):
    kgml_path = pathway['kgml_path']
    file_path = app.config['KGML_FOLDER']+"/"+ kgml_path.split('\\')[0]+"/"+kgml_path.split('\\')[1]

    edges, all_pathway_cids = parse_kgml(file_path)

    valid_cids = pathway_cids

    sub_edges = [(src, dst) for src, dst in edges if src in valid_cids and dst in valid_cids]
    adj_matrix, cid_order = build_adjacency_matrix(valid_cids, sub_edges)
    chagmap = dict(achanges)
    changes = []
    for i in range(len(cid_order)):
        changes.append(chagmap.get(cid_order[i], 0))
    # changes = achanges[list(set(achanges.index) & set(cid_order))].values
    piv = calculate_piv(adj_matrix, changes)
    return piv


def plot_bubble_chart(results):
    """
    绘制气泡图，只显示 PIV>0 且 -log(p) >= 0.5 的通路。
    气泡大小随 PIV 增大而增大，颜色从浅黄到深红随 -log(p) 增大而加深。
    """
    # 过滤条件
    df = results[(results['piv'] > 0) & (results['log_p'] >= 0.5)].copy()
    if df.empty:
        raise ValueError("没有满足 PIV>0 且 -log(p)≥0.5 的通路。")

    # 归一化 -log(p) 到 [0,1]，用于颜色映射
    logp = df['log_p'].values
    norm_logp = (logp - logp.min()) / (logp.max() - logp.min())

    # 气泡大小：PIV 放大尺度
    piv = df['piv'].values
    sizes = 500 * (piv / piv.max())  # 最大 PIV 对应 size=100

    # 颜色映射：使用黄到红的调色板
    cmap = cm.get_cmap('YlOrRd')
    colors = cmap(norm_logp)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(df['piv'], df['log_p'],
                     s=sizes,
                     c=colors,
                     alpha=0.7,
                     edgecolors='white',
                     linewidths=0.5)

    # 添加标签
    for _, row in df.iterrows():
        plt.annotate(
            row['pathway_id'],
            (row['piv'], row['log_p']),
            textcoords="offset points",
            xytext=(0, 8),
            ha='center',
            fontsize=8
        )

    plt.xlabel('PIV (Pathway Impact Value)')
    plt.ylabel('-log(p)')
    plt.title('Pathway Enrichment Analysis')
    plt.grid(True)

    # colorbar 显示 -log(p) 对应颜色
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=logp.min(), vmax=logp.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('-log(p) (color scale)')

    # 保存并编码为 base64
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png', dpi=150)
    buffer.seek(0)
    plot_image = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plot_image


@app.route('/pathway/<int:id>', methods=['GET', 'POST'])
def pathway(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + 'fin' + '.csv')
    data = pd.read_csv(file_path)
    group_col = data.columns[1]  # 组别所在列www
    groups = sorted(data[group_col].unique().tolist())
    tmp_folder = os.path.join(UPLOAD_FOLDER, 'analysis_tmp/', f'{secure_filename(file_info.file_name).split(".")[0]}/')
    os.makedirs(tmp_folder, exist_ok=True)

    if request.method == 'POST':
        # 获取前端传递的参数
        species = request.form.get('species')
        group1 = request.form.get('group1')
        group2 = request.form.get('group2')
        tmp_path = os.path.join(UPLOAD_FOLDER, 'analysis_tmp/',
                                f'{secure_filename(file_info.file_name).split(".")[0]}/',
                                f'pathway_{group1 + "VS" + group2}.csv')

        # 加载浓度数据
        concentration_data = load_concentration_data(file_path)

        # 加载通路数据
        pathway_data = load_pathway_data(species)

        if pathway_data is None or len(pathway_data) == 0:
            return jsonify({'error': 'No pathway data available for this species'}), 400

        # 执行富集分析
        results = global_test_enrichment(pathway_data, concentration_data, group1, group2)

        results.sort_values(by='piv', ascending=False, inplace=True)

        # 保存结果
        results.to_csv(tmp_path, index=False)

        # 绘制气泡图
        plot_image = plot_bubble_chart(results)

        return render_template('pathway_results.html',
                               results=results.to_dict('records'),
                               plot_image=plot_image,
                               id=id)

    # GET请求，显示表单
    return render_template('pathway_form.html', id=id, groups=groups, species_list={'hsa': 'Human', 'mmu': 'Mouse'})


# 路由1：展示分析结果列表页面
@app.route('/analysis_results/<int:id>')
def analysis_results(id):
    files = []
    file_info = FileInfo.query.filter_by(id=id).first()
    tmp_folder = os.path.join(UPLOAD_FOLDER, 'analysis_tmp/', f'{secure_filename(file_info.file_name).split(".")[0]}/')
    if os.path.exists(tmp_folder):
        for filename in os.listdir(tmp_folder):
            if filename.endswith(".csv"):
                if filename.lower().startswith("anova"):
                    method = "ANOVA"
                    group_info = "无"
                elif filename.lower().startswith("plsda_"):
                    method = "PLS-DA"
                    base_name = filename[len("plsda_"):-len(".csv")]
                    group_info = base_name.replace("VS", " vs ")
                elif filename.lower().startswith("oplsda_"):
                    method = "OPLS-DA"
                    base_name = filename[len("oplsda_"):-len(".csv")]
                    group_info = base_name.replace("VS", " vs ")
                elif filename.lower().startswith("pathway_"):
                    method = "代谢通路分析"
                    base_name = filename[len("pathway_"):-len(".csv")]
                    group_info = base_name.replace("VS", " vs ")
                elif filename.lower().startswith("ttest"):
                    method = "T-Test"
                    base_name = filename[len("ttest_"):-len(".csv")]
                    group_info = base_name.replace("VS", " vs ")
                else:
                    method = "未知"
                    group_info = ""
                files.append({
                    "filename": filename,
                    "method": method,
                    "group_info": group_info
                })
    return render_template("analysis_results.html", id=id, files=files)


@app.route('/generate_instruction/<int:id>/<language>', methods=['POST'])
def generate_instruction(id, language):
    selected_files = request.form.getlist("selected_files")
    file_info = FileInfo.query.filter_by(id=id).first()

    # Base sentences in both languages
    sentences = {
        'zh': {
            'sentence1': f"我正在进行{file_info.context}的研究。研究对象为{file_info.sample}，样本分组和每组样本数量为{file_info.gp}。研究目的是{file_info.purpose}",
            'sentence2_prefix': "通过一维NMR氢谱指认出的代谢物有：",
            'sentence2_suffix': "。",
            'sentence2_empty': "通过一维NMR氢谱指认出的代谢物信息暂无。",
            'oplsda': "OPLS-DA分析，在{}的分析中，VIP值大于1的代谢物有：",
            'plsda': "PLS-DA分析，在{}的分析中，VIP值大于1的代谢物有：",
            'anova': "ANOVA分析，p小于0.05的代谢物有：",
            'ttest': "T-Test分析，在{}的分析中，p小于0.05的代谢物有：",
            'pathway': "代谢通路分析，在{}的分析中,p值小于0.05的代谢通路有：",
            'final': "基于以上分析结果，帮我编写研究报告。模仿我给出的QA例子的分析逻辑和行文思路，特别时在初步结论中，结合代谢物所处环境和生物学意义，以及代谢物参与的代谢途径，分析目前初步的实验结果，并给出进一步研究的计划。(用中文回答)",
            'fold_change_note': "（注意fold_change<1说明{}相对于{}的代谢物水平下调，fold_change>1则说明{}相对于{}的代谢物水平上调。）"
        },
        'en': {
            'sentence1': f"I am conducting research on {file_info.context}. The research subjects are {file_info.sample}, with sample groups and quantities as {file_info.gp}. The research purpose is {file_info.purpose}",
            'sentence2_prefix': "Metabolites identified by 1D NMR hydrogen spectroscopy include: ",
            'sentence2_suffix': ".",
            'sentence2_empty': "No metabolite information identified by 1D NMR hydrogen spectroscopy is currently available.",
            'oplsda': "OPLS-DA analysis, in the comparison between {}, metabolites with VIP values greater than 1 include: ",
            'plsda': "PLS-DA analysis, in the comparison between {}, metabolites with VIP values greater than 1 include: ",
            'anova': "ANOVA analysis, metabolites with p-values less than 0.05 include: ",
            'ttest': "T-Test analysis, in the comparison between {}, metabolites with p-values less than 0.05 include: ",
            'pathway': "pathway analysis, in the comparison between {}, metabolic pathways with p_value less than 0.05 include: ",
            'final': "Based on the above analysis results, please help me prepare a research report. Follow the analytical logic and writing style of the QA examples I provided, especially in the preliminary conclusions. Combine the environmental and biological significance of the metabolites, as well as the metabolic pathways they participate in, to analyze the current preliminary experimental results and propose further research plans.(Use English)",
            'fold_change_note': "(Note: fold_change<1 indicates that {} has downregulated metabolite levels relative to {}, while fold_change>1 indicates that {} has upregulated metabolite levels relative to {}.)"
        }
    }

    lang = sentences.get(language, 'zh')

    sentence1 = lang['sentence1']

    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file_info.file_name) + 'fin' + '.csv')
    tmp_folder = os.path.join(UPLOAD_FOLDER, 'analysis_tmp/', f'{secure_filename(file_info.file_name).split(".")[0]}/')

    # Read metabolite concentration file
    if os.path.exists(file_path):
        df_met = pd.read_csv(file_path)
        metabolite_names = list(df_met.columns[2:])
        sentence2 = lang['sentence2_prefix'] + "、".join(metabolite_names) + lang['sentence2_suffix']
    else:
        sentence2 = lang['sentence2_empty']

    # Generate analysis description sentences
    analysis_sentences = []
    for filename in selected_files:
        filepath = os.path.join(tmp_folder, filename)
        if not os.path.exists(filepath):
            continue

        base_name = os.path.splitext(filename)[0]
        if filename.lower().startswith("oplsda_"):
            groups = base_name[len("oplsda_"):].split("VS")
            group_text = f"{groups[0]} and {groups[1]}" if len(groups) >= 2 else ""
            df_oplsda = pd.read_csv(filepath)
            if "VIP值" in df_oplsda.columns and "代谢物" in df_oplsda.columns:
                mets = df_oplsda[df_oplsda["VIP值"] > 1]["代谢物"].tolist()
                sentence = lang['oplsda'].format(group_text) + "、".join(mets) + lang['sentence2_suffix']
                analysis_sentences.append(sentence)

        elif filename.lower().startswith("plsda_"):
            groups = base_name[len("plsda_"):].split("VS")
            group_text = f"{groups[0]} and {groups[1]}" if len(groups) >= 2 else ""
            df_plsda = pd.read_csv(filepath)
            if "VIP值" in df_plsda.columns and "代谢物" in df_plsda.columns:
                mets = df_plsda[df_plsda["VIP值"] > 1]["代谢物"].tolist()
                sentence = lang['plsda'].format(group_text) + "、".join(mets) + lang['sentence2_suffix']
                analysis_sentences.append(sentence)

        elif filename.lower().startswith("anova"):
            df_anova = pd.read_csv(filepath)
            if "p_value" in df_anova.columns and "代谢物" in df_anova.columns and "F_stat" in df_anova.columns and "effect_size" in df_anova.columns:
                df_filtered = df_anova[df_anova["p_value"] < 0.05]
                details = []
                for idx, row in df_filtered.iterrows():
                    if language == 'zh':
                        details.append(f"{row['代谢物']}(F={row['F_stat']:.4f},效应量={row['effect_size']:.4f})")
                    else:
                        details.append(f"{row['代谢物']}(F={row['F_stat']:.4f},effect size={row['effect_size']:.4f})")
                sentence = lang['anova'] + "；".join(details) + lang['sentence2_suffix']
                analysis_sentences.append(sentence)

        elif filename.lower().startswith("ttest"):
            groups = base_name[len("ttest_"):].split("VS")
            group_text = f"{groups[0]} and {groups[1]}" if len(groups) >= 2 else ""
            df_ttest = pd.read_csv(filepath)
            if "fold_change" in df_ttest.columns and "代谢物" in df_ttest.columns and "p_value" in df_ttest.columns:
                df_filtered = df_ttest[df_ttest["p_value"] < 0.05]
                details = []
                for idx, row in df_filtered.iterrows():
                    if language == 'zh':
                        details.append(
                            f"{row['代谢物']}，fold_change={row['fold_change']:.4f}，p_value={row['p_value']:.4f}")
                    else:
                        details.append(
                            f"{row['代谢物']}, fold_change={row['fold_change']:.4f}, p_value={row['p_value']:.4f}")
                sentence = lang['ttest'].format(group_text) + "；".join(details)
                if len(groups) >= 2:
                    sentence += lang['fold_change_note'].format(groups[1], groups[0], groups[1], groups[0])
                analysis_sentences.append(sentence)

        elif filename.lower().startswith("pathway_"):
            groups = base_name[len("pathway_"):].split("VS")
            group_text = f"{groups[0]} and {groups[1]}" if len(groups) >= 2 else ""
            df_pathway = pd.read_csv(filepath)
            if "pathway_name" in df_pathway.columns and "p_value" in df_pathway.columns and "piv" in df_pathway.columns and "match_status" in df_pathway.columns:
                df_filtered = df_pathway[df_pathway["p_value"] < 0.05]
                pathway_name = df_filtered["pathway_name"].tolist()
                sentence = lang['pathway'].format(group_text) + "、".join(pathway_name) + lang['sentence2_suffix']
                analysis_sentences.append(sentence)

    sentence3 = " ".join(analysis_sentences)
    sentence_final = lang['final']

    instruction = "\n".join([sentence1, sentence2, sentence3, sentence_final])

    # Save instruction as TXT file
    txt_filename = f"instruction_{id}.txt"
    txt_filepath = os.path.join(tmp_folder, txt_filename)
    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write(instruction)

    return render_template("instruction.html", id=id, instruction=instruction, selected_files=selected_files,
                           language=language)


@app.route('/llm/<int:id>')
def llm(id):
    # 渲染问答页面
    return render_template('llm.html', id=id)


@app.route('/llm_stream/<int:id>')
def llm_stream(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    tmp_folder = os.path.join(UPLOAD_FOLDER, 'analysis_tmp/', f'{secure_filename(file_info.file_name).split(".")[0]}/')
    txt_filename = f"instruction_{id}.txt"
    txt_filepath = os.path.join(tmp_folder, txt_filename)

    def generate():
        # 读取本地 question.txt 文件中的提问内容
        with open('question.txt', 'r', encoding='utf-8') as f:
            question = f.read()
        with open(txt_filepath, 'r', encoding='utf-8') as f:
            question1 = f.read()
        question = question + question1

        reasoning_content = ""
        final_content = ""
        is_answering = False

        # 发送初始消息
        yield "data: [开始流式输出]\n\n"

        stream = client.chat.completions.create(
            model="deepseek-r1",  # 根据实际情况选择模型名称
            messages=[{"role": "user", "content": question}],
            stream=True
        )

        # 遍历流式返回的每个 chunk
        for chunk in stream:
            if not getattr(chunk, 'choices', None):
                continue
            delta = chunk.choices[0].delta

            # 推理过程部分
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                text = delta.reasoning_content
                reasoning_content += text
                yield f"event: reasoning\ndata: {text}\n\n"
            # 最终回答部分
            elif hasattr(delta, 'content') and delta.content:
                if not is_answering:
                    yield "event: final\ndata: [模型开始回答]\n\n"
                    is_answering = True
                text = delta.content
                final_content += text
                yield f"event: final\ndata: {text}\n\n"
        # 流式结束后发送结束标识
        yield "event: final\ndata: [回答完成]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route('/qianwen/<int:id>')
def qianwen(id):
    # 渲染问答页面（显示加载框）
    return render_template("qianwen.html", id=id)


@app.route('/qianwen_answer/<int:id>')
def qianwen_answer(id):
    file_info = FileInfo.query.filter_by(id=id).first()
    tmp_folder = os.path.join(UPLOAD_FOLDER, 'analysis_tmp/', f'{secure_filename(file_info.file_name).split(".")[0]}/')
    txt_filename = f"instruction_{id}.txt"
    txt_filepath = os.path.join(tmp_folder, txt_filename)

    with open('question.txt', 'r', encoding='utf-8') as f:
        question = f.read()
    with open(txt_filepath, 'r', encoding='utf-8') as f:
        question1 = f.read()
    question = question + question1

    # 调用大模型接口（非流式模式）
    completion = client.chat.completions.create(
        model="qwen-plus",  # 可根据实际情况更换模型名称
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': question}
        ],
    )
    # 将返回的结果转换为 JSON 字符串后解析为字典
    response_json = json.loads(completion.model_dump_json())
    answer = ""
    # 根据返回格式，提取 choices[0]["message"]["content"]
    if "choices" in response_json and len(response_json["choices"]) > 0:
        message = response_json["choices"][0].get("message", {})
        answer = message.get("content", "")
    return jsonify({"answer": answer})


def get_status(file_name):
    file_path2 = os.path.join(UPLOAD_FOLDER, secure_filename(file_name) + '_output' + '.csv')
    file_path3 = os.path.join(UPLOAD_FOLDER, secure_filename(file_name) + '_quantification' + '.csv')
    file_path4 = os.path.join(UPLOAD_FOLDER, secure_filename(file_name) + '_quantregion' + '.csv')
    status = []
    sta = ''
    if os.path.exists(file_path2):
        status.append('已指认')
    if os.path.exists(file_path3):
        status.append('已定量')
    if os.path.exists(file_path4):
        status.append('已划分积分区域')
    if len(status) > 0:
        for i in status:
            sta = sta + i + ';'
    return sta


@app.route('/progress')
def get_progress():
    return jsonify({'progress': progress})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
