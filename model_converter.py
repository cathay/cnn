from tensorflow import lite

converter = lite.TFLiteConverter.from_keras_model_file('/home/san.tran/eroad/innovation/eroad_action_detection_resnet_10_epoch.h5')
tfmodel = converter.convert()
open("model.tflite","wb").write(tfmodel)