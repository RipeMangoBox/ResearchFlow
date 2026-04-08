import os

def export_pdf_structure(root_dir, output_file):
    # 记录找到的文件数量
    count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # os.walk 会递归遍历文件夹
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    # 获取相对路径（去除根目录部分）
                    relative_path = os.path.relpath(os.path.join(root, file), root_dir)
                    
                    # 将系统路径分隔符替换为 |
                    # 使用 os.sep 处理跨平台兼容性
                    formatted_path = relative_path.replace(os.sep, '|')
                    
                    f.write(formatted_path + '\n')
                    count += 1
    
    print(f"处理完成！共找到 {count} 个 PDF 文件，结果已存入 {output_file}")

# --- 配置区域 ---
target_folder = 'paperPDFs'  # 替换为你的论文根目录
output_txt = 'paperPDFs/paper_list.txt'           # 输出文件名

export_pdf_structure(target_folder, output_txt)