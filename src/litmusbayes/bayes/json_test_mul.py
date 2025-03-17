import json

path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector2.jsonl"


def load_stacked_json_arrays(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        # 一次性读取所有内容（如果文件非常大，几个GB，需要换流式写法）
        content = f.read()

    decoder = json.JSONDecoder()
    pos = 0
    all_data = []

    while pos < len(content):
        # 跳过空白字符
        while pos < len(content) and content[pos].isspace():
            pos += 1

        if pos >= len(content):
            break

        try:
            # raw_decode 会从字符串开始处解析一个完整的 JSON 对象
            # 并返回 (解析出的对象, 这个对象结束的字符位置)
            obj, end_pos = decoder.raw_decode(content[pos:])

            # 因为你的顶层结构是列表 [...]，我们将内容 extend 进去
            # 如果 obj 本身就是单个对象而不是列表，请改用 all_data.append(obj)
            if isinstance(obj, list):
                all_data.extend(obj)
            else:
                all_data.append(obj)

            pos += end_pos
        except json.JSONDecodeError as e:
            print(f"❌ 解析错误发生在字符位置 {pos}: {e}")
            break

    return all_data


# --- 运行测试 ---
try:
    data = load_stacked_json_arrays(path)
    print(f"✅ 成功读取! 总共获取了 {len(data)} 条数据。")
    # 打印第一条看看对不对
    if data:
        print("第一条数据的 Name:", data[0].get('name'))
except Exception as e:
    print(f"读取失败: {e}")