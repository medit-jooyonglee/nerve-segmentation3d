# trainer-main
 * torch trainer using dynamic import and mlflow-tracking
 * [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet/tree/master) 저자 코드를 리팩토링하였음
 * 모델 추적 - mlflow tracking
   * ~~~ python
     
     uri = mlflow_path if mlflow_path.startswith('http') else f'file:///{mlflow_path}'
     mlflow.set_tracking_uri(uri=uri)
     
     with mlflow.start_run(run_name=self.run_name, experiment_id=self.experiment_id):
          flatten_config = flatten(self.config)
          for key, val in flatten_config.items():
               mlflow.log_params({key: val})
          .....
     
     ~~~
 * dynamic import
   * yaml 파일에 모델 종류, loaders, loss, metrics 등을 정의하고 동적으로 불러온다.
     * ~~~ yaml
       model:
         name: PIDNetWrapper
         param1: 100
         ....
       loaders:
         name: PIDNetDatasetWrapper
         ...
       loss:
         name: CrossEntropyLossSq
         ...
       metrics:
         name:
           [Precision]
         ...
     
       device: cuda:0
       ~~~
   * 클래스나 함수를 변수명을 입력받아 처리 importlib
     * 
       ~~~ python
          # 동적으로 class, functuion 등올 가져온다
          def get_class(class_name, modules):
              for module in modules:
                  try:
                      m = importlib.import_module(module)
                  except Exception as e:
                      logger = get_logger('error')
                      logger.error(e.args)
                      continue
                      # return None
                  clazz = getattr(m, class_name, None)
                  if clazz is not None:
                      return clazz
              raise RuntimeError(f'Unsupported class: {class_name}')
          ...
          # model class 가조오는 wrapper
          def get_model(model_config):
              model_class = get_class(model_config['name'], modules=[
                  'trainer.test.testmodel',
                  'interfaces.pidnetmodel',
                  'vit_pytorch.vit',
                  'relationattention',
                  # 'teethnet.models.unet3d.model',
                  # 'planning_gan.models.transunet.transunet3d.vit_seg_modeling',
              ])
              try:
                  return model_class(**model_config)
              except TypeError as e:
                  arg = ml_collections.ConfigDict(model_config)
                  return model_class(arg)
              except Exception as e:
                  raise ValueError(e.args)
              ...
          # configure 파일 설정해서 train / test 실행
          def main():
              filepath = os.path.join(os.path.dirname(__file__), 'configfiles/piednet_grouping_default_lightweight.yaml')
              config = trainer.load_config(filepath)
              build_train = trainer.create_trainer(config, trainer.__file__)
              build_train.fit()
    
       ~~~

 * 학습 모델, 데이터셋, loss등을 
   * config - dictionary v.s. config
     * dict - key & values
     * ml_collection - "dot-based access to fields" 
   * trainer : trainer.py::create_trainer & trainer.py::Trainer
     * configure 
       * model, loss, metrics, dataset 등을 설정 
       * yaml파일 dict로 읽어서 ml_collection으로 변환해서 처리
   
     * dataset : dataloader.py::get_train_loaders
     * models : modelloader.py::get_model
     * loss : loss.py::get_loss_criterion 
     * metrics : metrics.py::get_evaluation_metric
 