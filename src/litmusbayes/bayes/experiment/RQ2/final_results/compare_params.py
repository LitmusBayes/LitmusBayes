import json


def compare_json_files(file1_path, file2_path):
    # 读取两个 JSON 文件
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        with open(file2_path, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
    except FileNotFoundError as e:
        print(f"找不到文件: {e}")
        return

    # 找到两个文件中共有的 litmus test 名称 (取交集)
    # 假设你的 JSON 最外层是字典，键为 test 的名字
    common_tests = set(data1.keys()).intersection(set(data2.keys()))

    if not common_tests:
        print("两个文件没有共同的 litmus test，无法对比。")
        return

    different_tests_count = 0
    different_tests_names = []

    # 遍历所有共同的 test 进行对比
    for test_name in common_tests:
        param1 = data1[test_name].get("param", [])
        param2 = data2[test_name].get("param", [])

        # 判断两个数组是否完全一致
        if param1 != param2:
            different_tests_count += 1
            different_tests_names.append(test_name)
            print(test_name, param1, param2)

    # 打印统计结果
    print(f"文件1共有 {len(data1)} 条数据")
    print(f"文件2共有 {len(data2)} 条数据")
    print(f"两个文件共同包含 {len(common_tests)} 条 Litmus test")
    print("-" * 50)
    print(f"⭐ param 不同的 litmus test 总个数为: {different_tests_count}")
    print("-" * 50)

    # 如果想看看具体是哪几个不同，可以取消下面代码的注释
    if different_tests_count > 0:
        print("发生不同的 Litmus test 列表 (前10个示例):")
        for name in different_tests_names[:10]:  # 只打印前10个避免刷屏
            print(f" - {name}")
        if different_tests_count > 10:
            print(f"   ... 等共 {different_tests_count} 个")


if __name__ == "__main__":
    # 在这里填入你的两个 json 文件的实际路径
    file1 = "../best_params_recommendation_robust_final_kmeans_cross_10_factor_2_final_1.json"
    file2 = "../best_params_recommendation_robust_final_kmeans_cross_10_factor_2_final.json"

    compare_json_files(file1, file2)
