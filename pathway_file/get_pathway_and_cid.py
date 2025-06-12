import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm


def get_all_pathways(species):
    """
    获取指定物种所有通路信息。

    参数:
        species (str): 物种代码，如 "hsa"。

    返回:
        list of tuples: 每个元组包含 (pathway_id, pathway_name)。
    """
    url = f"http://rest.kegg.jp/list/pathway/{species}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"获取通路列表失败，状态码: {response.status_code}")

    pathways = []
    # 每行格式: "path:hsa01100\tPathway name; additional info"
    for line in response.text.strip().splitlines():
        if line:
            try:
                parts = line.split('\t')
                if len(parts) >= 2:
                    # 第一个字段类似 "path:hsa01100"，取后面的部分
                    pathway_id = parts[0]
                    # 通路名称可能包含 ";", 这里取第一个部分
                    pathway_name = parts[1].split(";")[0].strip()
                    pathways.append((pathway_id, pathway_name))
            except:
                continue
    return pathways


def download_kgml(pathway_id, save_dir="kgml_files"):
    """
    根据通路ID下载KGML文件，并保存到指定目录中。

    参数:
        pathway_id (str): 通路ID，如 "hsa01100"
        save_dir (str): KGML文件保存的目录，默认为 "kgml_files"

    返回:
        str: 保存的文件路径，如果下载失败，则返回空字符串。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    url = f"http://rest.kegg.jp/get/{pathway_id}/kgml"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"通路 {pathway_id} 的KGML文件下载失败，状态码: {response.status_code}")
        return ""

    file_path = os.path.join(save_dir, f"{pathway_id}.xml")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    return file_path


def parse_metabolite_cids_from_kgml(kgml_file):
    """
    解析KGML文件，提取其中所有代谢物的KEGG CID，并去掉前缀 "cpd:"。

    参数:
        kgml_file (str): KGML文件路径

    返回:
        list: 代谢物CID列表，例如 ["C00011", "C00014", "C00022"]
    """
    try:
        tree = ET.parse(kgml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"解析文件 {kgml_file} 失败: {e}")
        return []

    metabolite_ids = set()
    for entry in root.findall('entry'):
        # 检查entry节点类型为compound
        if entry.attrib.get('type') == 'compound':
            names = entry.attrib.get('name', '')
            # 名称字段可能有多个，以空格分隔，如 "cpd:C00011 cpd:C00014"
            for name in names.split():
                if name.startswith("cpd:"):
                    # 去掉前缀 "cpd:"
                    metabolite_ids.add(name.replace("cpd:", ""))
    return sorted(metabolite_ids)


def get_species_pathway_metabolites(species):
    """
    根据物种代码获取该物种所有通路的代谢物CID，同时保存KGML文件。

    参数:
        species (str): 物种代码，如 "hsa"

    返回:
        pandas.DataFrame: 包含以下列：
            - pathway_id: 通路ID（如 "hsa01100"）
            - pathway_name: 通路名称
            - metabolites: 逗号分隔的代谢物CID（如 "C00011,C00014,C00022"）
            - kgml_file: 保存的KGML文件路径
    """
    pathways = get_all_pathways(species)
    data = []
    if os.path.exists("mmu_pathways.csv"):
        outp = pd.read_csv("mmu_pathways.csv")
        c = len(outp)
    else:
        c=0

    for pathway_id, pathway_name in tqdm(pathways):
        if c >0:
            c=c-1
            continue
        else:
            print(f"处理通路: {pathway_id} - {pathway_name}")
            kgml_file = download_kgml(pathway_id)
            if kgml_file:
                metabolite_ids = parse_metabolite_cids_from_kgml(kgml_file)
                metabolites_str = ",".join(metabolite_ids)
            else:
                metabolites_str = ""

            data.append({
                "pathway_id": pathway_id,
                "pathway_name": pathway_name,
                "metabolites": metabolites_str,
                "kgml_file": kgml_file
            })
            pd.DataFrame(data, columns=["pathway_id", "pathway_name", "metabolites", "kgml_file"]).to_csv("mmu_pathways1.csv", index=False)

    df = pd.DataFrame(data, columns=["pathway_id", "pathway_name", "metabolites", "kgml_file"])
    return df


if __name__ == "__main__":
    # 示例：处理人类物种 "hsa"
    species = "mmu"
    df = get_species_pathway_metabolites(species)
    # 如果需要保存为CSV文件
    # df.to_csv("hsa_pathways.csv", index=False)
