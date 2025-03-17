asm_code = (
    "lw %[x9],0(%[x8])\\n"
    "#_litmus_P1_5\\n\t"
    "andi %[x10],%[x9],128\\n"
    "#_litmus_P1_6\\n\t"
    "add %[x13],%[x12],%[x10]\\n"
    "#_litmus_P1_7\\n\t"
    "lw %[x11],0(%[x13])\\n"
    "#_litmus_P1_8\\n"
)

with open("output.cpp", "w") as f:
    f.write(f'"{asm_code}"\n')
