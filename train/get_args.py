import argparse
import os

from argsbase import get_baseargs


def get_args(main_args, trial=None):

    parser = get_baseargs()

    # special settings
    parser.add_argument("--lambda1", type=float, default=1.0, help="weight of loss1")
    parser.add_argument("--lambda2", type=float, default=0.3, help="weight of loss2")
    parser.add_argument("--lambda3", type=float, default=0.1, help="weight of loss2")

    parser.add_argument("--scale", type=int, default=1, help="scale_mask_softmax")
    parser.add_argument("--N", type=int, default=1, help="num proxies per class")
    parser.add_argument("--r", type=float, default=0.5, help="ration which determines how many proxies are selected")

    parser.add_argument("--posmask", type=str, default="soft", help="posmask type in [soft, hard, none]")
    parser.add_argument("--use-rlp", type=bool, default=True, help="use rlp")
    parser.add_argument("--proxy-reg-type", type=str, default="bce", help="proxy reg type in [bce, l2, none]")


    args = parser.parse_args()
    args = argparse.Namespace(**vars(args), **vars(main_args))
    # args.batch_size = 32
    flag = ""
    if hasattr(args, "noise_rate") and args.noise_rate > 0:
        flag += f"noise_rate{args.noise_rate}"
    if not args.use_rlp:
        flag += "w/o_rlp"
    if args.proxy_reg_type != "bce":
        flag += f"_{args.proxy_reg_type}"

    args.method = "ProxyGML"

    if flag == "":
        args.save_dir = os.path.join(args.save_dir, args.method, args.dataset, str(args.output_dim))
    else:
        args.save_dir = os.path.join(args.save_dir, args.method, flag, args.dataset, str(args.output_dim))
    return args
