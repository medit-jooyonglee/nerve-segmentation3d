import argparse
from trainer import get_model, create_trainer, load_config

def main():
    parser = argparse.ArgumentParser(description='ianseg')
    parser.add_argument('--config',default='configure/roi_segmentation_miccai.yaml', type=str)
    args = parser.parse_args()
    config = load_config(args.config, '')
     
    trainer = create_trainer(config)
    trainer.fit()


if __name__ == '__main__':
    main()
