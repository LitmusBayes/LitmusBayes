#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple

# -------------------------
# 数据结构
# -------------------------

class Var:
    ATOMIC_INT = "ATOMIC_INT"
    VOLATILE_INT = "VOLATILE_INT"
    PLAIN_INT = "PLAIN_INT"

    def __init__(self, name: str, init: int = 0, kind: str = PLAIN_INT):
        self.name = name
        self.init = int(init)
        self.kind = kind  # ATOMIC_INT / VOLATILE_INT / PLAIN_INT

class ThreadBlock:
    def __init__(self, name: str):
        self.name = name  # P0 / P1 / ...
        self.body: List[str] = []  # 原始 C 行（稍后做轻量改写）
        self.param_kinds: Dict[str, str] = {}  # 变量名 -> kind

class Program:
    def __init__(self):
        self.vars: Dict[str, Var] = {}        # 全局变量
        self.threads: List[ThreadBlock] = []  # 线程列表
        self.exists_raw: str = ""             # exists(...) 内的表达式
        self.iters: int = 100000              # 默认迭代次数

    def merge_kinds_from_params(self):
        # 依据线程参数类型修正变量声明类型（优先级：ATOMIC > VOLATILE > PLAIN）
        for tb in self.threads:
            for name, kind in tb.param_kinds.items():
                v = self.vars.get(name)
                if v is None:
                    v = Var(name, 0, kind)
                    self.vars[name] = v
                else:
                    if v.kind == Var.ATOMIC_INT:
                        continue
                    if kind == Var.ATOMIC_INT:
                        v.kind = Var.ATOMIC_INT
                        continue
                    if v.kind == Var.VOLATILE_INT:
                        continue
                    if kind == Var.VOLATILE_INT:
                        v.kind = Var.VOLATILE_INT

# -------------------------
# 解析器
# -------------------------

def parse_file(text: str) -> Program:
    lines = text.splitlines()
    i, n = 0, len(lines)

    def skip_blank(idx: int) -> int:
        while idx < n and lines[idx].strip() == "":
            idx += 1
        return idx

    p = Program()

    # 可选首行：C <title>
    i = skip_blank(i)
    if i < n and lines[i].lstrip().startswith("C "):
        i += 1

    # 初值块：{ [x] = 0; [y] = 0; ... }
    i = skip_blank(i)
    if i >= n or "{" not in lines[i]:
        raise SyntaxError("缺少初值块：应为 { [x] = 0; [y] = 0; ... }")
    init_block, i = collect_brace_block(lines, i)
    parse_init_block(init_block, p)

    # 多个线程块：P0 (params) { ... }
    PHEAD = re.compile(r'^\s*(P\d+)\s*\((.*)\)\s*\{\s*$')
    while True:
        i = skip_blank(i)
        if i >= n:
            break
        if lines[i].strip().startswith("exists"):
            break
        m = PHEAD.match(lines[i])
        if not m:
            raise SyntaxError(f"期待线程头：Pn (params) {{ ，但读到：{lines[i]}")
        tb = ThreadBlock(m.group(1))
        parse_param_list(m.group(2), tb, p)
        i += 1
        body_block, i = collect_body_until_matching_brace(lines, i)
        tb.body = body_block
        p.threads.append(tb)

    p.merge_kinds_from_params()

    # exists(...) 行
    i = skip_blank(i)
    if i >= n or not lines[i].strip().startswith("exists"):
        raise SyntaxError("缺少 exists(...) 行")
    p.exists_raw = extract_exists(lines[i].strip())

    return p

def collect_brace_block(lines: List[str], i: int) -> Tuple[List[str], int]:
    """收集从含 '{' 的行开始到匹配的 '}' 为止（包含首尾）。"""
    block = []
    depth = 0
    n = len(lines)
    while i < n:
        s = lines[i]
        if "{" in s:
            depth += s.count("{")
        block.append(s)
        if "}" in s:
            depth -= s.count("}")
            if depth <= 0:
                i += 1
                break
        i += 1
    if depth != 0:
        raise SyntaxError("花括号不匹配（初值块）")
    return block, i

def collect_body_until_matching_brace(lines: List[str], i: int) -> Tuple[List[str], int]:
    """在线程头之后，从下一行开始，收集到与头部 '{' 匹配的 '}' 为止（不含头行）。"""
    body = []
    depth = 1  # 头部已有一个 '{'
    n = len(lines)
    while i < n:
        s = lines[i]
        depth += s.count("{")
        if "}" in s:
            depth -= s.count("}")
            if depth == 0:
                i += 1
                break
            else:
                # 本行有 '}', 但仍在块内，去掉 '}' 再保存
                body.append(s)
        else:
            body.append(s)
        i += 1
    if depth != 0:
        raise SyntaxError("花括号不匹配（线程体）")
    return body, i

def parse_init_block(block: List[str], p: Program):
    # 形如： { [x] = 0; [y] = 0; }
    ITEM = re.compile(r'.*\[\s*([A-Za-z_]\w*)\s*]\s*=\s*(-?\d+)\s*;.*')
    for line in block:
        m = ITEM.match(line)
        if m:
            name, val = m.group(1), int(m.group(2))
            p.vars[name] = Var(name, val, Var.PLAIN_INT)

def parse_param_list(plist: str, tb: ThreadBlock, p: Program):
    # 例："atomic_int* x, atomic_int* y, atomic_int* zero" 或 "atomic_int* x, volatile int* y"
    parts = [s.strip() for s in plist.split(",") if s.strip()]
    for part in parts:
        # 最后一个标识符视为变量名
        name = re.sub(r'[*\s]+', ' ', part).split()[-1]
        kind = kind_from_param(part)
        tb.param_kinds[name] = kind
        if name not in p.vars:
            p.vars[name] = Var(name, 0, kind)

def kind_from_param(s: str) -> str:
    t = s.replace("*", " ").strip()
    if t.startswith("atomic_int"):
        return Var.ATOMIC_INT
    if t.startswith("volatile"):
        return Var.VOLATILE_INT
    return Var.PLAIN_INT

def extract_exists(line: str) -> str:
    # exists(x=1 /\ y=1)
    m = re.match(r'^\s*exists\s*\((.*)\)\s*$', line)
    if not m:
        raise SyntaxError(f"无法解析 exists 行：{line}")
    return m.group(1).strip()

# -------------------------
# 代码生成
# -------------------------

def gen_c(p: Program) -> str:
    out = []
    w = out.append

    w('#include <stdio.h>')
    w('#include <stdlib.h>')
    w('#include <pthread.h>')
    w('#include <stdatomic.h>')
    w('#include <string.h>')
    w('')

    # 全局变量
    for v in p.vars.values():
        if v.kind == Var.ATOMIC_INT:
            w(f'_Atomic int {v.name} = {v.init};')
        elif v.kind == Var.VOLATILE_INT:
            w(f'volatile int {v.name} = {v.init};')
        else:
            w(f'int {v.name} = {v.init};')
    w('')

    # 线程函数
    for tb in p.threads:
        w(f'void* thread_{tb.name}(void* _){{')
        for raw in tb.body:
            s = rewrite_body_line(raw, p)
            if s.strip() == "":
                continue
            w(f'  {s}')
        w('  return NULL;')
        w('}')
        w('')

    # 重置函数
    w('static inline void reset_globals(){')
    for v in p.vars.values():
        if v.kind == Var.ATOMIC_INT:
            w(f'  atomic_store_explicit(&{v.name}, {v.init}, memory_order_relaxed);')
        else:
            w(f'  {v.name} = {v.init};')
    w('}')
    w('')

    # exists 检查（最终状态）
    w('static inline int check_exists(){')
    for v in p.vars.values():
        if v.kind == Var.ATOMIC_INT:
            w(f'  int {v.name}_snap = atomic_load_explicit(&{v.name}, memory_order_relaxed);')
        else:
            w(f'  int {v.name}_snap = {v.name};')
    ex = rewrite_exists_expr(p.exists_raw, set(p.vars.keys()))
    w(f'  return ({ex}) ? 1 : 0;')
    w('}')
    w('')

    # main
    w('int main(int argc, char** argv){')
    w(f'  long iters = {p.iters};')
    w('  long count = 0;')
    w(f'  pthread_t tids[{len(p.threads)}];')
    w('  for (long it=0; it<iters; ++it){')
    w('    reset_globals();')
    for idx, tb in enumerate(p.threads):
        w(f'    pthread_create(&tids[{idx}], NULL, thread_{tb.name}, NULL);')
    for idx, _ in enumerate(p.threads):
        w(f'    pthread_join(tids[{idx}], NULL);')
    w('    count += check_exists();')
    w('  }')
    w('  printf("exists observed: %ld / %ld\\n", count, iters);')
    w('  return 0;')
    w('}')
    w('')
    return "\n".join(out)

def rewrite_body_line(line: str, p: Program) -> str:
    """把线程体里的 *x / *y 解引用写法，宽松改写为全局标识符直接使用；
       atomic_* 调用保持原样。"""
    s = line.strip()
    if not s:
        return ""
    # 把 "*name" 替换为 "name"（仅替换独立 token）
    for v in p.vars.keys():
        s = re.sub(rf'\*\s*\b{re.escape(v)}\b', v, s)
    return s

def rewrite_exists_expr(ex: str, varnames: set) -> str:
    # 把 /\ -> &&, \/ -> ||, 单独的 '=' -> '=='
    s = ex
    s = re.sub(r'\/\\', '&&', s)  # /\ -> &&
    s = re.sub(r'\\\/', '||', s)  # \/ -> ||

    s = s.replace('==', '@@EQ@@')
    s = s.replace('=', '==')
    s = s.replace('@@EQ@@', '==')

    # 变量替换为 var_snap
    for v in sorted(varnames, key=lambda x: -len(x)):  # 先替换长名字避免前缀冲突
        s = re.sub(rf'\b{re.escape(v)}\b', f'{v}_snap', s)
    return s

# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser(
        description="将 herd/C 风格（子集）的 litmus 测试转换为可编译 C 程序（stdatomic.h + pthread）")
    ap.add_argument("input", type=Path, help="输入 .litmus 文件")
    ap.add_argument("output", type=Path, help="输出 .c 文件")
    ap.add_argument("--iters", type=int, default=100000, help="运行迭代次数（默认 100000）")

    args = ap.parse_args()
    text = args.input.read_text(encoding="utf-8")
    prog = parse_file(text)
    prog.iters = args.iters
    csrc = gen_c(prog)
    args.output.write_text(csrc, encoding="utf-8")
    print(f"生成完成: {args.output}")

if __name__ == "__main__":
    main()
