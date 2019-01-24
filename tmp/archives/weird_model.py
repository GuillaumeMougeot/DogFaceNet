import tensorflow as tf

out_num = 125
def net(input_shape, emb_size=4):
    input_image = tf.keras.Input(input_shape,name='image_input')
    
    x = tf.keras.layers.Conv2D(32, (3,3), padding='same')(input_image)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Concatenate()([input_image]*11 + [x])
    
    x = tf.keras.layers.MaxPooling2D((3,3))(x)
    
    im = tf.keras.layers.MaxPooling2D((3,3))(input_image)
    
    r = tf.keras.layers.Dropout(0.25)(x)
    
    x = tf.keras.layers.Conv2D(128, (3,3), padding='same')(r)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Concatenate()([im]*21 + [r,x])
    
    x = tf.keras.layers.MaxPooling2D((3,3))(x)
    
    im = tf.keras.layers.MaxPooling2D((3,3))(im)
    
    r = tf.keras.layers.Dropout(0.25)(x)
    
    x = tf.keras.layers.Conv2D(256, (3,3), padding='same')(r)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Concatenate()([r,x])
    
    x = tf.keras.layers.MaxPooling2D((3,3))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
#     emb = tf.keras.layers.Dense(10, activity_regularizer='l2')(x)
#     out = tf.keras.layers.Dense(out_num, kernel_regularizer='l2', use_bias=False, name='output')(emb)
    
    out = tf.keras.layers.Dense(out_num, activation='softmax', name='output')(x)
    
    model = tf.keras.Model(inputs=[input_image], outputs=out)
    return model