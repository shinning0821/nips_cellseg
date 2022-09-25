import argparse

def return_args():
    parser = argparse.ArgumentParser("Baseline for Microscopy image segmentation")
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="./data/Train_Pre_3class/",
        type=str,
        help="training data path; subfolders: images, labels",
    )
    parser.add_argument(
        "--ssl_data_path",
        default="./data/Train_unlable_3class/",
        type=str,
        help="unlabled training data path;",
    )
    parser.add_argument(
        "--work_dir", default="./baseline/work_dir", help="path where to save models and logs"
    )
    parser.add_argument("--seed", default=2022, type=int)
    # parser.add_argument("--resume", default=False, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=4, type=int)

    # Model parameters
    parser.add_argument(
        "--model_name", default="cross_pseudo", help="select mode: unet, unetr, swinunetr"
    )
    parser.add_argument("--num_class", default=3, type=int, help="segmentation classes")
    parser.add_argument(
        "--input_size", default=256, type=int, help="segmentation classes"
    )
    # Training parameters
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--epoch_tolerance", default=100, type=int)
    parser.add_argument("--initial_lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--model_path", default='/data112/wzy/NIPS/baseline/work_dir/unet_4_3class', help="learning rate")
    args = parser.parse_args()
    return args