from src.litmusbayes.bayes.litmus_params import LitmusParams

if __name__ == "__main__":
    litmus_name = "LB"
    litmus_path = f"/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive/{litmus_name}.litmus"
    litmus_dir = f'/home/whq/Desktop/code_list/perple_test/stride/{litmus_name}'
    params = LitmusParams()
    params.apply_standard_form()
    params.set_riscv_gcc()
    params.append_by_dict({"stride":31})
    print(params.to_dict())
    print(params.to_litmus7_format(litmus_path, litmus_dir))