import os
from pathlib import Path


def process_files_and_record(folder_paths):
    target_extensions = {".pdf", ".md"}

    md_log_path = "md_paths.txt"
    pdf_log_path = "pdf_paths.txt"

    open(md_log_path, "w", encoding="utf-8").close()
    open(pdf_log_path, "w", encoding="utf-8").close()

    for folder_num, folder_path in enumerate(folder_paths, 1):
        root_path = Path(folder_path)

        if not root_path.exists() or not root_path.is_dir():
            print(f"⚠️ 警告: 路径不存在或不是文件夹: {root_path}")
            continue

        print(f"--- 正在处理文件夹 {folder_num}: {root_path.name} ---")

        total_md = 0
        total_pdf = 0
        modified_count = 0
        ignored_count = 0

        for file_path in root_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in target_extensions:
                if "skill" in file_path.name.lower():
                    ignored_count += 1
                    continue

                current_file = file_path
                if "-" in file_path.name:
                    new_name = file_path.name.replace("-", "_")
                    new_file_path = file_path.with_name(new_name)
                    try:
                        file_path.rename(new_file_path)
                        current_file = new_file_path
                        modified_count += 1
                    except Exception as e:
                        print(f"❌ 重命名失败 '{file_path.name}': {e}")

                clean_path = str(current_file.relative_to(root_path).with_suffix(""))

                if current_file.suffix.lower() == ".md":
                    with open(md_log_path, "a", encoding="utf-8") as f:
                        f.write(clean_path + "\n")
                    total_md += 1
                elif current_file.suffix.lower() == ".pdf":
                    with open(pdf_log_path, "a", encoding="utf-8") as f:
                        f.write(clean_path + "\n")
                    total_pdf += 1

        print(f"文件夹 {root_path.name} 处理完毕：")
        print(f"  - 重命名文件数: {modified_count}")
        print(f"  - 忽略(含skill)数: {ignored_count}")
        print(f"  - 记录 MD 路径数: {total_md}")
        print(f"  - 记录 PDF 路径数: {total_pdf}")
        print("-" * 50)

    print("\n✅ 处理完成。")


if __name__ == "__main__":
    my_folders = [
        "./paperAnalysis",
        "./paperPDFs",
    ]

    process_files_and_record(my_folders)

