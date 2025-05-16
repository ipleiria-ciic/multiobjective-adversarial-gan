import os
import json
import argparse
from solver import Solver
from datetime import datetime
from torch.backends import cudnn
from data_loader import get_loader, get_loader_class

def main(config):
    cudnn.benchmark = True

    loader, class_loader = None, None
    dataset_mean = (0.485, 0.456, 0.406)
    dataset_std = (0.229, 0.224, 0.225)

    loader = get_loader(
        config.dataset,
        config.dataset_train,
        config.dataset_test,
        dataset_mean,
        dataset_std,
        config.crop_size,
        config.image_size,
        config.batch_size,
        config.mode,
        config.num_workers)

    class_loader = get_loader_class(
        config.dataset,
        config.dataset_train,
        dataset_mean,
        dataset_std,
        config.crop_size,
        config.image_size,
        config.batch_size,
        config.mode,
        config.num_workers)
    
    solver = Solver(loader, class_loader, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument('--c_dim', type=int, default=10)
    parser.add_argument('--crop_size', type=int, default=178)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=32)
    parser.add_argument('--d_conv_dim', type=int, default=32)
    parser.add_argument('--c_conv_dim', type=int, default=32) 
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--c_repeat_num', type=int, default=6)     
    parser.add_argument('--lambda_cls', type=float, default=0.25)  
    parser.add_argument('--lambda_rec', type=float, default=1.3)
    parser.add_argument('--lambda_gp', type=float, default=1)
    parser.add_argument('--lambda_perturbation', type=float, default=0.9)
    parser.add_argument('--nadir_slack', type=float, default=1.1) # Can range between 1.1 and 1.05.
    parser.add_argument('--disc_weights', nargs='+', default=['0', '1'])
                                            
    # Training configuration
    parser.add_argument('--dataset', type=str, default='Imagewoof')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_iters', type=int, default=1000000)
    parser.add_argument('--num_iters_decay', type=int, default=100)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--c_lr', type=float, default=0.00012)      
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--c_beta1', type=float, default=0.9)
    parser.add_argument('--resume_iters', type=int, default=None)  

    # Test configuration
    parser.add_argument('--test_iters', type=int, default=500)

    # Miscellaneous
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', default=False)

    # Directories
    parser.add_argument('--dataset_train', type=str, default='Dataset/Imagewoof/train')
    parser.add_argument('--dataset_test', type=str, default='Dataset/Imagewoof/test')
    parser.add_argument('--delta', type=float)
    parser.add_argument('--attack', type=str)
    parser.add_argument('--log_dir', type=str, default='SuperstarGAN/logs')
    parser.add_argument('--model_save_dir', type=str, default='SuperstarGAN/models')
    parser.add_argument('--sample_dir', type=str, default='SuperstarGAN/samples')
    parser.add_argument('--result_dir', type=str, default='SuperstarGAN/results')
    
    # Step size
    parser.add_argument('--log_step', type=int, default=5)
    parser.add_argument('--sample_step', type=int, default=5000)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--lr_update_step', type=int, default=500)

    config = parser.parse_args()
    config_dict = vars(config)

    log_data = {
        "training_session": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **config_dict
    }

    os.makedirs(config.log_dir, exist_ok=True)
    log_file = os.path.join(config.log_dir, "arguments_log.json")

    if os.path.exists(log_file):
        with open(log_file, "r+") as log_arg:
            try:
                data = json.load(log_arg)
            except json.JSONDecodeError:
                data = []
            log_data["id"] = len(data) + 1
            data.append(log_data) 
            log_arg.seek(0)
            json.dump(data, log_arg, indent=4)
    else:
        log_data["id"] = 1
        with open(log_file, "w") as log_arg:
            json.dump([log_data], log_arg, indent=4)

    print("[ INFO ] Parameters saved into '{}'.".format(log_file))
    
    main(config)