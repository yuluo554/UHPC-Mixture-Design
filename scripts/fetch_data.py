# -*- coding: utf-8 -*-
"""
fetch_data.py — 按 docs/DATA.md 下载并校验 UHPC 数据集

用法:
    python scripts/fetch_data.py --list            # 只列出清单，不下载
    python scripts/fetch_data.py                   # 下载全部自动可下的数据集
    python scripts/fetch_data.py --only global_uhpc scc_zenodo
    python scripts/fetch_data.py --verify-only     # 只校验本地已有文件的 SHA256
    python scripts/fetch_data.py --force           # 覆盖已存在但校验失败的文件

设计原则（见 docs/DATA.md）:
  - data/raw/ 不入库；本脚本是唯一的补齐入口
  - 每个文件都会校验 SHA256；不匹配就报错退出（不会静默跳过）
  - 需要登录/人工点击的数据集（Kaggle、Figshare）明确标记为 manual，
    打印下载指令后以退出码 2 结束
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# 与 data/column_mapping.py 的 DATASET_CONFIGS 一一对应
# dest 路径相对于 data/，必须与 DATASET_CONFIGS[...]["path"] 完全一致
DATASETS: dict[str, dict] = {
    "literature": {
        "dest": "literature.csv",
        "sha256": None,          # 仓库自带，不需要下载
        "note": "项目原生 37 行文献数据，已入库",
        "bundled": True,
    },
    "bolbolvand_comp": {
        "dest": "raw/bolbolvand_uhpc/uhpc_compressive_strength.csv",
        "url": "https://raw.githubusercontent.com/MiladBolbolvand/UHPC-Machine-Learning/main/uhpc_compressive_strength_1.csv",
        "sha256": "b22c2daa8b83921ec886daadc6e3ca132590eb3da706f3456987a669b2569c40",
        "size": 52591,
        "source": "https://github.com/MiladBolbolvand/UHPC-Machine-Learning",
        "note": "Bolbolvand UHPFRC 抗压强度（863 行）。仓库原始文件名为 "
                "uhpc_compressive_strength_1.csv，落盘时改名为 DATASET_CONFIGS 期望的名字",
    },
    "bolbolvand_flex": {
        "dest": "raw/bolbolvand_uhpc/uhpc_flexural_strength.csv",
        "url": "https://raw.githubusercontent.com/MiladBolbolvand/UHPC-Machine-Learning/main/uhpc_flexural_strength_1.csv",
        "sha256": "ccab44a420a2dadbf862593f12400decb8490214337e6e92f090442fefc8a7cf",
        "size": 20271,
        "source": "https://github.com/MiladBolbolvand/UHPC-Machine-Learning",
        "note": "Bolbolvand UHPFRC 抗弯强度（321 行）",
    },
    "sean_jiao": {
        "dest": "raw/sean_jiao_uhpc/Bayesian Neural Networks/抗压强度数据.xlsx",
        "url": "https://raw.githubusercontent.com/sean-jiao/UHPC-prediction/main/UHPC-pred.zip",
        "zip_member": "Bayesian Neural Networks/抗压强度数据.xlsx",
        "sha256": "b52b06e48d5fbe0a93060edd09476004d3b1c20995b8e32b3472559739f9e510",
        "size": 70777,
        "source": "https://github.com/sean-jiao/UHPC-prediction",
        "note": "从 UHPC-pred.zip 中取出该成员；zip 整体另有 SHA256 "
                "9531a1773d64055ee5b8d946f7c454007f0a5410b923a9be8c8e67cfb458b943",
    },
    "global_uhpc": {
        "dest": "raw/global_uhpc/Global Dataset of UHPC Mix Designs with Supplement/UHPC Dataset  (Version-2).xlsx",
        "url": "https://data.mendeley.com/public-files/datasets/czb7ww5pkz/files/e16412ff-f805-4f67-8be3-dd6920e5362d/file_downloaded",
        "sha256": "4274da215fd1884a8c9be46f1459c2e1dff44b52ad34b9a529fda75a5c077d3a",
        "size": 7067833,
        "source": "https://data.mendeley.com/datasets/czb7ww5pkz/2 (DOI 10.17632/czb7ww5pkz.2, CC BY 4.0)",
        "note": "Mendeley 全球 UHPC 库（2188 组配比）。文件名里有两个空格，落盘时必须一致",
    },
    "scc_zenodo": {
        "dest": "raw/scc_zenodo/SCC Dataset on Mixture Design & Key Properties - v1.csv",
        "url": "https://zenodo.org/api/records/10569517/files/SCC%20Dataset%20on%20Mixture%20Design%20%26%20Key%20Properties%20-%20v1.csv/content",
        "sha256": "71629f3ba531cc2d236abaf9921bc943a0e187fdc74710d2e57b8e401efb981a",
        "size": 597963,
        "source": "https://zenodo.org/records/10569517 (DOI 10.5281/zenodo.10569517, CC BY 4.0)",
        "note": "自密实混凝土库（2504 行）。按 latin-1 + sep=';' 读取",
    },
    "steelfiber_mendeley": {
        "dest": "raw/steelfiber_mendeley/characterizacion and slump test.xlsx",
        "url": "https://data.mendeley.com/public-files/datasets/22y4cxbsd8/files/ed0af447-3e19-469a-baeb-25a16d6f1853/file_downloaded",
        "sha256": "06a89175f36119b1490977cff161c42dae4a803e59d33d056808e3c438685e5a",
        "size": 11181,
        "source": "https://data.mendeley.com/datasets/22y4cxbsd8/3 (DOI 10.17632/22y4cxbsd8.3, CC BY 4.0)",
        "note": "钢纤维混凝土材料表征（Slump 6 行 + 骨料 5 行）",
    },
    "umn_uhpc": {
        "dest": "raw/umn_uhpc/UHPC NDE Dataset.xlsx",
        "url": "https://conservancy.umn.edu/server/api/core/bitstreams/34ed3a86-bc5a-4fce-9db9-f37a3ecfd1a9/content",
        "sha256": "5ceb566051d2845a9169a6f873eb7be86475951892eb9113b64046ec1fd94f48",
        "size": 90803,
        "source": "https://conservancy.umn.edu/handle/11299/272536 (DOI 10.13020/h3m2-1h76)",
        "note": "UMN UHPC 抗压 + UPV + 回弹（108 行）",
    },
    "kaggle_steelfiber": {
        "dest": "raw/kaggle_steelfiber/database_en.xlsx",
        "sha256": "209b41ad3f486c9f2084b2691e5d3480bcab340231b02759cff62d34c52c89fc",
        "size": 112717,
        "manual": True,
        "source": "https://www.kaggle.com/datasets/arthurmurta/steel-fiber-reinforced-concreteann",
        "note": "Kaggle 需要账号登录。下载后把 database_en.xlsx 放到 dest 路径；"
                "已记录 SHA256，可用 --verify-only 校验",
    },
    "figshare_recycled": {
        "dest": "raw/figshare_recycled/Dataset.csv",
        "sha256": "4deeb02f656752f1f0dcbb3c4b94cd72510408323c4648f40010d0f5486d9325",
        "size": 16491,
        "manual": True,
        "source": "https://tandf.figshare.com/articles/dataset/31376593",
        "note": "Figshare 有 WAF/登录限制，脚本无法直连。手动下载 Dataset.csv 后放到 dest 路径",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "UHPC-Mixture-Design/fetch_data"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_one(key: str, force: bool = False, verify_only: bool = False) -> str:
    """返回 ok / skipped / manual / failed"""
    cfg = DATASETS[key]
    dest = DATA_DIR / cfg["dest"]

    if cfg.get("bundled"):
        print(f"[skip] {key}: 仓库自带 {cfg['dest']}")
        return "skipped"

    if cfg.get("manual") and not verify_only:
        print(f"[manual] {key}: {cfg['source']}")
        print(f"         请手动下载并放到 data/{cfg['dest']}")
        print(f"         期望 SHA256: {cfg['sha256']}")
        return "manual"

    if dest.exists():
        digest = sha256_file(dest)
        if digest == cfg["sha256"]:
            print(f"[ok]     {key}: 已存在且校验通过 ({digest[:12]}…)")
            return "skipped"
        if not force:
            print(f"[FAIL]   {key}: 文件已存在但 SHA256 不匹配")
            print(f"         实际 {digest}")
            print(f"         期望 {cfg['sha256']}")
            print("         确认要覆盖请加 --force")
            return "failed"
        print(f"[warn]   {key}: 现有文件校验失败，--force 覆盖")

    if verify_only:
        print(f"[FAIL]   {key}: 文件不存在 ({dest})")
        return "failed"

    if not cfg.get("url"):
        print(f"[FAIL]   {key}: 既没有直链也不是仓库自带，清单配置有误")
        return "failed"

    print(f"[get]    {key}: {cfg['url']}")
    try:
        payload = download(cfg["url"])
        if cfg.get("zip_member"):
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                payload = zf.read(cfg["zip_member"])
    except Exception as exc:  # noqa: BLE001 - 这里就是要报出真实原因
        print(f"[FAIL]   {key}: 下载失败: {type(exc).__name__}: {exc}")
        return "failed"

    digest = hashlib.sha256(payload).hexdigest()
    if digest != cfg["sha256"]:
        print(f"[FAIL]   {key}: 下载内容 SHA256 不匹配，已丢弃")
        print(f"         实际 {digest}")
        print(f"         期望 {cfg['sha256']}")
        return "failed"

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    print(f"[ok]     {key}: 写入 data/{cfg['dest']} ({len(payload)} bytes)")
    return "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载并校验 UHPC 数据集")
    parser.add_argument("--only", nargs="*", help="只处理指定数据集 key")
    parser.add_argument("--list", action="store_true", help="只列出清单")
    parser.add_argument("--verify-only", action="store_true", help="只校验本地文件")
    parser.add_argument("--force", action="store_true", help="覆盖校验失败的已有文件")
    args = parser.parse_args(argv)

    keys = args.only or list(DATASETS)
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        print(f"未知数据集: {unknown}")
        print(f"可用: {list(DATASETS)}")
        return 1

    if args.list:
        for k in keys:
            cfg = DATASETS[k]
            kind = "bundled" if cfg.get("bundled") else ("manual" if cfg.get("manual") else "auto")
            print(f"{k:24s} {kind:8s} {cfg['dest']}")
        return 0

    results = {k: fetch_one(k, force=args.force, verify_only=args.verify_only) for k in keys}

    failed = [k for k, r in results.items() if r == "failed"]
    manual = [k for k, r in results.items() if r == "manual"]
    ok = [k for k, r in results.items() if r in ("ok", "skipped")]

    print()
    print(f"通过 {len(ok)} | 需手动 {len(manual)} | 失败 {len(failed)}")
    if failed:
        print(f"失败: {failed}")
        return 1
    if manual:
        print(f"需手动下载: {manual}（其余数据集已就绪）")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
