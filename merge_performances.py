#!/usr/bin/env python3
"""
自动合并 performances 目录下的结果
- 检测 outputs/default 下最新的文件夹
- 合并所有子文件夹中的 db_data 到 merge_results/db_data
- 合并所有 *_details.jsonl 文件为 merged_details.jsonl
"""

import os
import shutil
import glob
from pathlib import Path
from typing import List


def find_latest_output_dir(base_dir: str = "outputs/default") -> Path:
    """查找 outputs/default 下最新的文件夹"""
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"目录不存在: {base_dir}")
    
    # 获取所有子目录
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"在 {base_dir} 下没有找到子文件夹")
    
    # 按修改时间排序，返回最新的
    latest_dir = max(subdirs, key=lambda p: p.stat().st_mtime)
    print(f"找到最新的目录: {latest_dir}")
    return latest_dir


def merge_db_data(performances_dir: Path, merge_results_dir: Path):
    """合并所有子文件夹中的 db_data 到 merge_results/db_data"""
    merge_db_data_dir = merge_results_dir / "db_data"
    merge_db_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历所有子文件夹（排除 merge_results）
    for subdir in performances_dir.iterdir():
        if not subdir.is_dir() or subdir.name == "merge_results":
            continue
        
        db_data_dir = subdir / "db_data"
        if not db_data_dir.exists():
            print(f"警告: {subdir.name} 中没有找到 db_data 目录，跳过")
            continue
        
        print(f"正在合并 {subdir.name}/db_data 到 merge_results/db_data...")
        
        # 复制 db_data 中的所有文件
        for item in db_data_dir.iterdir():
            dest_path = merge_db_data_dir / item.name
            
            if item.is_file():
                # 如果是文件，直接复制（如果已存在则覆盖）
                shutil.copy2(item, dest_path)
                print(f"  复制文件: {item.name}")
            elif item.is_dir():
                # 如果是目录，递归复制
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(item, dest_path)
                print(f"  复制目录: {item.name}")


def merge_details_jsonl(performances_dir: Path, merge_results_dir: Path):
    """合并所有 *_details.jsonl 文件为 merged_details.jsonl"""
    merge_results_dir.mkdir(parents=True, exist_ok=True)
    output_file = merge_results_dir / "merged_details.jsonl"
    
    # 查找所有 *_details.jsonl 文件（排除 merge_results 目录）
    details_files = []
    for subdir in performances_dir.iterdir():
        if not subdir.is_dir() or subdir.name == "merge_results":
            continue
        
        # 查找该目录下的所有 *_details.jsonl 文件
        pattern = str(subdir / "*_details.jsonl")
        found_files = glob.glob(pattern)
        details_files.extend(found_files)
    
    if not details_files:
        print("警告: 没有找到任何 *_details.jsonl 文件")
        return
    
    print(f"找到 {len(details_files)} 个 *_details.jsonl 文件:")
    for f in details_files:
        print(f"  - {f}")
    
    # 合并所有文件
    print(f"正在合并到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for details_file in sorted(details_files):
            print(f"  处理: {details_file}")
            with open(details_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if line:  # 跳过空行
                        outfile.write(line + '\n')
    
    print(f"合并完成: {output_file}")


def main():
    """主函数"""
    try:
        # 1. 找到最新的输出目录
        latest_dir = find_latest_output_dir()
        performances_dir = latest_dir / "performances"
        
        if not performances_dir.exists():
            raise ValueError(f"目录不存在: {performances_dir}")
        
        # 2. 创建 merge_results 目录
        merge_results_dir = performances_dir / "merge_results"
        merge_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"创建/使用合并目录: {merge_results_dir}")
        
        # 3. 合并 db_data
        print("\n=== 开始合并 db_data ===")
        merge_db_data(performances_dir, merge_results_dir)
        
        # 4. 合并 details.jsonl 文件
        print("\n=== 开始合并 *_details.jsonl 文件 ===")
        merge_details_jsonl(performances_dir, merge_results_dir)
        
        print("\n=== 合并完成 ===")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

