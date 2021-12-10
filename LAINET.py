import os, sys, time
from tqdm.auto import tqdm
import yaml
import argparse

from lainet.training import train_main
from lainet.inference import inference_main


#Example: python3 LAINET.py -r /scratch/users/dmasmont/LAI/seven/founders.vcf -m /scratch/users/dmasmont/LAI/seven/founders.map -q /scratch/users/dmasmont/LAI/seven/all_val.vcf -o /scratch/users/dmasmont/LAI/output_lainet/seven_bk -i 20 -g /scratch/users/dmasmont/LAI/world_wide_references/allchrs.b37.gmap

if __name__ == "__main__":
    tic = time.time()
    
    print('Welcome to LAI-Net')
    
    parser = argparse.ArgumentParser(description="""LAI-Net command interface ---
    Example (Training+Inference): python3 LAINET.py -i 20 -q test.vcf -r founders.vcf -m founders.map -o output/test- -d "cuda" ---
    Example (Inference Only): python3 LAINET.py -q test.vcf -n output/pre-trained- -o output/test-""")
    
    parser.add_argument('-q', '--query', type=str, help='Absolute path of query samples .VCF file (Required always)')
    parser.add_argument('-r', '--reference', type=str, help='Absolute path of reference panel .VCF file (Required for training)')
    parser.add_argument('-m', '--map', type=str, help='Absolute path of labels list .map file (Required for training)')
    parser.add_argument('-g', '--gmap', type=str, help='Genetic map .gmap file - only used when saving results (Optional)')
    parser.add_argument('-o', '--output', type=str, help='Output folder and prefix (Required always)')
    parser.add_argument('-n', '--network-folder', type=str, help='Folder and prefix of trained network (Required for inference only)')
    parser.add_argument('-c', '--config', type=str, default='configs/default.yaml', help='Config .yaml file (Optional)')
    parser.add_argument('-d', '--device', type=str, help='Device to train and inference network (Optional)')
    parser.add_argument('-i', '--chm', type=str, help='Chromosome Number (Required for training)')

    args = parser.parse_args()
    
    print('Running LAI-Net with the following arguments: {}'.format(args))

    # Check input keywords
    if args.query is None:
        print('A .vcf query file is required. Use -h for more info')
        sys.exit(0)
    
    # Load config file
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    # Overwrite flags into config file
    if args.device is not None:
        config['TRAINING']['DEVICE'] = args.device
        
        
    print('LAI-Net configuration file is... {}'.format(config))
    print('-----------------------------------------------------------------')
    
    # If a network is provided - only run inference
    is_inference_only = (args.network_folder is not None)
    
    if not is_inference_only:
        train_main(config, args.reference, args.map, args.output, chm=args.chm, genetic_map_path=args.gmap)
        args.network_folder = args.output
        
    inference_main(config, args.query, args.network_folder, args.output)
    
    print('Done after... {} seconds!'.format(time.time()-tic))