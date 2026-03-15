import tensorflow as tf

src = r"E:\zju\AtuomaticScaling\DIAS\result\FPN_resnet50_Adam\models\STEP_158000.model"
dst = r"E:\zju\AtuomaticScaling\DIAS\result\FPN_resnet50_Adam\models\STEP_158000_keras3.h5"

print("Loading old SavedModel:", src)
model = tf.keras.models.load_model(src, compile=False)
print("Saving as H5:", dst)
model.save(dst, include_optimizer=False)