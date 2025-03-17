import os
import json


def main():
    # ================= 配置区域 =================
    # 1. Litmus 文件所在的文件夹路径
    folder_path = "/home/whq/Desktop/code_list/slide/src/slide/bayes/make_new_litmus/litmus_output"

    # 2. 包含有效 name 的 JSON 文件路径
    json_path = "/src/slide/bayes/make_new_litmus/litmus_vector_new_graph_filter.jsonl"

    # 3. 安全开关 (False = 只打印不删除 / True = 真的删除)
    # 建议先用 False 运行一次，确认打印出的文件确实是你想要删掉的
    ENABLE_DELETE = True
    # ===========================================

    print(f"正在读取 JSON 文件: {json_path} ...")

    valid_names = set()
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # 判断 JSON 是列表格式还是单个对象格式
            if isinstance(data, list):
                for item in data:
                    if 'name' in item:
                        valid_names.add(item['name'])
            elif isinstance(data, dict):
                # 即使只有一条数据的情况
                if 'name' in data:
                    valid_names.add(data['name'])
            else:
                print("Error: JSON 格式不符合预期 (不是 list 也不是 dict)")
                return

        print(f"JSON 读取成功，白名单中共有 {len(valid_names)} 个名字。")

    except Exception as e:
        print(f"读取 JSON 失败: {e}")
        return

    print(f"正在扫描文件夹: {folder_path} ...")

    files_to_delete = []
    kept_count = 0

    # 遍历文件夹
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".litmus"):
                # 获取不带后缀的文件名，例如 "Rfe.litmus" -> "Rfe"
                file_name_no_ext = os.path.splitext(file)[0]

                # 如果文件名不在白名单里，加入删除列表
                if file_name_no_ext not in valid_names:
                    full_path = os.path.join(root, file)
                    files_to_delete.append(full_path)
                else:
                    kept_count += 1

    # 输出结果
    print("-" * 50)
    print(f"扫描结束。")
    print(f"  - 在 JSON 中找到并保留的文件数: {kept_count}")
    print(f"  - JSON 中不存在 (即待删除) 的文件数: {len(files_to_delete)}")
    print("-" * 50)

    if not files_to_delete:
        print("没有文件需要删除。")
        return

    # 根据开关执行逻辑
    if not ENABLE_DELETE:
        print("【安全模式】(ENABLE_DELETE = False)")
        print("以下文件将被删除（但现在只是打印出来，并未实际执行删除）：")
        for p in files_to_delete:
            print(f"  [待删] {p}")
        print("\n提示：请检查上述列表。确认无误后，将脚本中的 ENABLE_DELETE 改为 True 即可执行删除。")

    else:
        print("【警告】(ENABLE_DELETE = True)")
        print("正在执行删除操作...")
        deleted_count = 0
        for p in files_to_delete:
            try:
                os.remove(p)
                print(f"  [已删除] {p}")
                deleted_count += 1
            except Exception as e:
                print(f"  [删除失败] {p}: {e}")

        print(f"\n操作完成，共删除了 {deleted_count} 个文件。")


if __name__ == "__main__":
    main()