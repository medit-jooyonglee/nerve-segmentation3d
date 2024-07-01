from trainer.trainer import create_evaluator, load_config


def main():
    config_file = 'configure/roi_detection_miccai.yaml'
    config = load_config(config_file, '')
    evaluator = create_evaluator(config)

    evaluator.validate()


if __name__ == '__main__':
    main()