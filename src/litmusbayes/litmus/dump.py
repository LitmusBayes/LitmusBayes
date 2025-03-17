

import os
from typing import *


def dump_define(name, value):
    return f'#define {name} {value}\n'


def dump_params(size, runs, avail, stride, timeloop, mode):
    params = [
        dump_define('SIZE_OF_TEST', size),
        dump_define('NUMBER_OF_RUN', runs),
        dump_define('AVAIL', avail),
        dump_define('STRIDE', stride),
        dump_define('MAX_LOOP', timeloop),
        # dump_define('MODE', mode)
    ]
    return ''.join(params)


def dump_template(filename: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f'{current_dir}/template/{filename}', 'r') as file:
        contents = file.read()
    return contents


def dump_ctx_t(shared_vars: List, out_regs: List):
    dump_ctx_var = lambda x: f'  int *{x};'
    dump_ctx_reg = lambda x: f'  int *out_{x[0]}_{x[1]};'
    template = dump_template('3_context.c')
    return template.replace('@VARS@', '\n'.join(map(dump_ctx_var, shared_vars))) \
        .replace('@REGS@', '\n'.join(map(dump_ctx_reg, out_regs)))


def dump_outcome_collection(shared_vars: List, out_regs: List):
    dump_out_var = lambda x, i: f'  static const int {x}_f = {i};'
    dump_out_reg = lambda x, i: f'  static const int out_{x[0]}_{x[1]}_f = {i};'
    dump_var_fmt = lambda x: f'{x}=%i'
    dump_reg_fmt = lambda x: f'{x[0]}:{x[1]}=%i'
    dump_var_arg = lambda x: f'(int)o[{x}_f]'
    dump_reg_arg = lambda x: f'(int)o[out_{x[0]}_{x[1]}_f]'

    outcome = dump_template('4_outcome.c')

    out_indices = []
    i = 0
    for reg in out_regs:
        out_indices.append(dump_out_reg(reg, i))
        i += 1
    for var in shared_vars:
        out_indices.append(dump_out_var(var, i))
        i += 1
    out_indices = '\n'.join(out_indices)
    nouts = str(len(shared_vars + out_regs))
    out_fmt = '; '.join(list(map(dump_reg_fmt, out_regs)) + list(map(dump_var_fmt, shared_vars)))
    out_args = ','.join(list(map(dump_reg_arg, out_regs)) + list(map(dump_var_arg, shared_vars)))

    replacements = {
        '@NOUTS@': nouts,
        '@OUT_INDICES@': out_indices,
        '@OUT_FMT@': out_fmt,
        '@OUT_ARGS@': out_args
    }

    for key, value in replacements.items():
        outcome = outcome.replace(key, value)
    return outcome


def dump_prefetch(name: str, shared_vars: List):
    dump_init_var = lambda x: f'  int *{x} = _a->{x};'
    dump_check_var = lambda x: f'    if (rand_bit(&(_a->seed)) && {x}[_i] != 0) fatal("{name}, check_globals failed");'
    prefetch = dump_template('5_prefetch.c')
    init_globals = '\n'.join(list(map(dump_init_var, shared_vars)))
    check_globals = '\n'.join(list(map(dump_check_var, shared_vars)))
    return prefetch.replace('@INIT_GLOBALS@', init_globals).replace('@CHECK_GLOBALS@', check_globals)


def dump_thread_code(name: str, pid: int, prog, in_regs, out_regs, trashed_regs):
    code = dump_template('7_litmuscode.c')
    # TODO: prog?


def dump_context(shared_vars: List, out_regs: List):
    context = dump_template('8_context.c')
    dump_malloc_check = lambda x: f'_a->{x} = malloc_check(size_of_test * sizeof(*(_a->{x})));'
    dump_malloc_check_var = dump_malloc_check
    dump_malloc_check_reg = lambda x: dump_malloc_check(f'out_{x[0]}_{x[1]}')
    dump_malloc_check_copy_var = lambda x: dump_malloc_check(f'cpy_{x}[_p]')

    dump_free = lambda x: f'free((void *)_a->{x});'
    dump_free_var = dump_free
    dump_free_reg = lambda x: dump_free(f'out_{x[0]}_{x[1]}')
    dump_free_copy_var = lambda x: dump_free(f'cpy_{x}[_p]')

    dump_reinit_var = lambda x: f'_a->{x}[_i] = 0;'
    dump_reinit_reg = lambda x: f'_a->out_{x[0]}_{x[1]}[_i] = -239487;'

    replacements = {
        '@INIT_REG@': '\n'.join(map(dump_malloc_check_reg, out_regs)),
        '@INIT_VAR@': '\n'.join(map(dump_malloc_check_var, shared_vars)),
        '@INIT_COPY_VAR@': '\n'.join(map(dump_malloc_check_copy_var, shared_vars)),
        '@FREE_REG@': '\n'.join(map(dump_free_reg, out_regs)),
        '@FREE_VAR@': '\n'.join(map(dump_free_var, shared_vars)),
        '@FREE_COPY_VAR@': '\n'.join(map(dump_free_copy_var, shared_vars)),
        '@REINIT_VAR@': '\n'.join(map(dump_reinit_var, shared_vars)),
        '@REINIT_REG@': '\n'.join(map(dump_reinit_reg, out_regs))
    }

    for key, value in replacements.items():
        context = context.replace(key, value)

    return context


def dump_zyva(name: str, np: int, shared_vars: List, out_regs: List):
    zyva = dump_template('9_zyva.c')
    dump_ps = lambda x: f'&P{x}'
    dump_read_reg = lambda x: f'      int _out_{x[0]}_{x[1]}_i = ctx.out_{x[0]}_{x[1]}[_i];'
    dump_read_var = lambda x: f'      int _{x}_i = ctx.{x}[_i];'
    dump_check_cpy_var = lambda x: \
        '      for (int _p = N-1 ; _p >= 0 ; _p--) {\n' + \
        f'        if (_x_i != ctx.cpy_{x}[_p][_i]) fatal("{name}, global {x} unstabilized") ;\n' + \
        '      }'
    dump_write_reg = lambda x: f'      o[out_{x[0]}_{x[1]}_f] = _out_{x[0]}_{x[1]}_i;'
    dump_write_var = lambda x: f'      o[{x}_f] = _{x}_i;'
    dump_cond_reg = lambda x: f'_out_{x[0]}_{x[1]}_i'
    dump_cond_var = lambda x: f'_{x}_i'

    replacements = {
        '@PS@': ','.join(map(dump_ps, range(np))),
        '@READ_REGS@': '\n'.join(map(dump_read_reg, out_regs)),
        '@READ_VARS@': '\n'.join(map(dump_read_var, shared_vars)),
        '@CHECK_CPY_VARS@': '\n'.join(map(dump_check_cpy_var, shared_vars)),
        '@COND_ARGS@': ','.join(list(map(dump_cond_reg, out_regs)) + list(map(dump_cond_var, shared_vars))),
        '@WRITE_REGS@': '\n'.join(map(dump_write_reg, out_regs)),
        '@WRITE_VARS@': '\n'.join(map(dump_write_var, shared_vars)),
    }

    for key, value in replacements.items():
        zyva = zyva.replace(key, value)

    return zyva


def dump_assout(litmus):
    # TODO
    pass


def dump_postlude(name: str, cond: str):
    # TODO: Hash?
    postlude = dump_template('11_postlude.c')
    return postlude.replace('@NAME@', name).replace('@COND@', cond)


def dump_func_name(name: str):
    sep_replacements = {
        '+': '_2B_',
        '.': '_2E_',
        '[': '_5B_',
        ']': '_5D_',
        '-': '_2D_'
    }

    func_name = name
    for key, value in sep_replacements.items():
        func_name = func_name.replace(key, value)

    if func_name[0].isdigit():
        func_name = 'X' + func_name

    return func_name


def dump_run(name: str):
    run = dump_template('12_run.c')
    return run.replace('@NAME@', name).replace('@FUNC_NAME@', dump_func_name(name))
