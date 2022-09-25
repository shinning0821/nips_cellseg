import argparse

def return_args():
    parser = argparse.ArgumentParser("Baseline for Microscopy image segmentation")
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/data112/wzy/NIPS/data/fluorescence/Pre",
        type=str,
        help="training data path; subfolders: images, labels",
    )
    parser.add_argument(
        "--ssl_data_path",
        default="/data112/wzy/NIPS/data/fluorescence/unlabel",
        type=str,
        help="unlabled training data path;",
    )
    parser.add_argument(
        "--work_dir", default="/data112/wzy/NIPS/baseline/work_dir", help="path where to save models and logs"
    )
    parser.add_argument("--seed", default=2022, type=int)
    # parser.add_argument("--resume", default=False, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=4, type=int)

    # Model parameters
    parser.add_argument(
        "--model_name", default="rdrop", help="select mode: unet, unetr, swinunetr"
    )
    parser.add_argument("--num_class", default=3, type=int, help="segmentation classes")
    parser.add_argument(
        "--input_size", default=256, type=int, help="segmentation classes"
    )
    # Training parameters
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--start_epoch", default=1, type=int)
    parser.add_argument("--epoch_tolerance", default=100, type=int)
    parser.add_argument("--initial_lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--model_path", default='/data112/wzy/NIPS/baseline/work_dir/rdrop_3class', help="checkpoint path")
    args = parser.parse_args()
    return args