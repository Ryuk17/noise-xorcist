import os


def main(args):
    if args.inf_scp:
        # 直接指定 scp 文件, 结果输出到 inf.scp 同级目录
        inf_scp = args.inf_scp
        ref_scp = args.ref_scp
        enh_folder = os.path.dirname(os.path.abspath(inf_scp))
    else:
        # 未指定 scp 时, 从 --enh_dir 拼接 inf.scp/ref.scp
        if not args.enh_dir:
            raise ValueError("请通过 --inf_scp/--ref_scp 或 --enh_dir 指定输入")
        enh_folder = args.enh_dir
        inf_scp = f'{enh_folder}/inf.scp'
        ref_scp = f'{enh_folder}/ref.scp'

    if args.metric == 'dnsmos':
        os.system(
            ('python ./evaluation/calculate_nonintrusive_dnsmos.py '
                f'--inf_scp {inf_scp} '
                f'--output_dir {enh_folder}/scoring_dnsmos '
                f'--device {args.device} '
                '--job 1 '
                '--convert_to_torch True '
                '--primary_model ./DNSMOS/DNSMOS/sig_bak_ovr.onnx '
                '--p808_model ./DNSMOS/DNSMOS/model_v8.onnx'
            )
        )
    elif args.metric == 'intrusive':
        os.system(
            ('python ./evaluation/calculate_intrusive_se_metrics.py '
             f'--ref_scp {ref_scp} '
             f'--inf_scp {inf_scp} '
             f'--output_dir {enh_folder}/scoring_intrusive '
             '--nj 8 '
             '--chunksize 1000'
            )
        )

    else:
        raise ValueError
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', required=True, help="Metric to be calculated")
    parser.add_argument('--inf_scp', default='', help='inf.scp 路径 (增强音频列表), 不指定则用 {enh_folder}/inf.scp')
    parser.add_argument('--ref_scp', default='', help='ref.scp 路径 (参考音频列表, intrusive 需要), 不指定则用 {enh_folder}/ref.scp')
    parser.add_argument('--enh_dir', default='', help='含 inf.scp/ref.scp 的目录, 不指定 --inf_scp 时从此目录拼接')
    parser.add_argument('--device', default='cpu', help='dnsmos 计算设备 (cpu/cuda)')

    args = parser.parse_args()
    main(args)
