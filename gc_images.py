"""
删除未在 Markdown 文件中引用的图片

该脚本会遍历指定目录下的所有 Markdown 文件，并检查其中的图片引用。
对于每张在 image_dir 下找到的图片，如果它没有在任何 .md 文件中出现，则会被删除。

图片格式支持：.png, .jpg, .gif, .webp

注意：
- 比较基于字符串的完全匹配，不处理 Markdown 中的链接是否指向正确的路径。
- 脚本会递归地搜索 markdown_dir 和 image_dir 目录。


使用方法：
    修改 markdown_dir 和 image_dir 变量为实际路径后运行脚本。
"""

import os
import glob
from pathlib import Path

# ========================= 配置参数 =========================
markdown_dir = Path(".")  # 指向你的 Markdown 文件夹路径
image_base_dir = Path("content")  # 指向图片的基础目录，将递归搜索其下的所有 image 目录
IMAGE_DIR_NAME = "images"
# ============================================================

# 支持识别的图片扩展名
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.gif", "*.webp")


def read_all_markdown_content(md_dir):
    """
    递归读取 markdown_dir 目录及其子目录下所有的 .md 文件，并合并成一个总的字符串。

    :param md_dir: markdown 文件夹路径
    :return: 所有 markdown 文件内容拼接后的大字符串
    """
    total_content = ""
    md_pattern = os.path.join(md_dir, "**", "*.md")
    for file_path in glob.glob(md_pattern, recursive=True):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                total_content += content
        except Exception as e:
            print(f"警告: 无法读取文件 {file_path}, 错误: {e}")
    return total_content


def find_all_image_dirs(base_dir):
    """
    递归查找 base_dir 下所有名为 'image' 的目录

    :param base_dir: 基础目录路径
    :return: 所有找到的 image 目录路径列表
    """
    image_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if IMAGE_DIR_NAME in dirs:
            image_dirs.append(Path(root) / IMAGE_DIR_NAME)
    return image_dirs


def delete_unreferenced_images(img_base_dir, full_md_content):
    """
    递归查找 img_base_dir 下所有名为 'image' 的目录，然后遍历这些目录下的图片，
    若其文件名未在 full_md_content 中完全匹配，则删除。

    :param img_base_dir: 图片基础目录路径
    :param full_md_content: 所有 Markdown 文件内容组成的大字符串
    """
    # 查找所有 image 目录
    image_dirs = find_all_image_dirs(img_base_dir)
    print(f"找到 {len(image_dirs)} 个 {IMAGE_DIR_NAME} 目录")

    for image_dir in image_dirs:
        print(f"扫描目录: {image_dir}")
        for ext in IMAGE_EXTENSIONS:
            pattern = os.path.join(image_dir, "**", ext)
            # 使用 glob 获取所有匹配的图片
            for img_path_str in glob.glob(pattern, recursive=True):
                img_path = Path(img_path_str)
                img_name = img_path.name  # 提取纯文件名，例如 'photo.jpg'

                # 在总的内容中查找该文件名是否被引用（完全匹配）
                if img_name not in full_md_content:
                    try:
                        os.remove(img_path)  # 删除文件
                        print(f"已删除未引用的图片: {img_path}")
                    except Exception as e:
                        print(f"警告: 无法删除 {img_path}, 错误: {e}")
                else:
                    print(f"保留图片 (已引用): {img_path}")


def main():
    """
    主函数：执行整个流程
    1. 先预加载所有 Markdown 文件内容
    2. 再递归查找所有 image 目录并判断图片是否需要删除
    """
    print("开始检查 Markdown 文件内容...")
    full_markdown_text = read_all_markdown_content(markdown_dir)
    print(f"已成功加载 {len(full_markdown_text)} 字符的 Markdown 内容")

    print("开始递归扫描并清理未引用的图片...")
    delete_unreferenced_images(image_base_dir, full_markdown_text)
    print("清理完成。")


if __name__ == "__main__":
    # 执行前再次确认路径设置正确
    assert markdown_dir != "/path/to/markdown/files", "请先修改 markdown_dir 配置项！"
    assert image_base_dir != "/path/to/images/folder", "请先修改 image_base_dir 配置项！"

    main()
