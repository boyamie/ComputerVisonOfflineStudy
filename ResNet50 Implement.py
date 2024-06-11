# ResNet50 Caltech101 dataset으로 implement

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Dense, MaxPooling2D, AveragePooling2D, Flatten, ZeroPadding2D

# resnet의 핵심인 residual block을 구성한다.
# input_data, ouput feature map의 수, layer 이름을 parameter로 받는다
def residual_block(input_tensor, num_filters, naming):
    x = Conv2D(kernel_size = (1, 1), filters = num_filters//4, kernel_initializer = 'he_normal', name=naming + '_a')(input_tensor)
    x = BatchNormalization(name = naming + '_BN' +  '_a')(x)
    x = Activation('relu')(x)

    x = Conv2D(kernel_size = (3, 3), filters = num_filters//4, kernel_initializer = 'he_normal', padding='same', name=naming + '_b')(x)
    x = BatchNormalization(name = naming + '_BN' +  '_b')(x)
    x = Activation('relu')(x)

    x = Conv2D(kernel_size = (1, 1), filters = num_filters, kernel_initializer = 'he_normal', name=naming + '_c')(x)
    x = BatchNormalization(name = naming + '_BN' +  '_c')(x)

    # input의 크기를 맞춰주기 위해
    x_ = Conv2D(kernel_size= (1, 1), filters = num_filters)(input_tensor)
    x_ = BatchNormalization()(x_)

    x = Add()([x, x_])

    x = Activation('relu')(x)

    return x

# 예측하려는 class의 개수를 받음
class MyResNet50(Model):
  def __init__(self, num_classes):
    super(MyResNet50, self).__init__()
    self.num_cnn_list=[3,4,6,3]
    self.channel_list=[256, 512, 1024, 2048]
    self._num_classes = num_classes


  def call(self, input_data):
    x = ZeroPadding2D(padding= (3, 3))(input_data)
    
    # conv1
    x = Conv2D(filters=64, kernel_size = (7,7), strides = 2, name='conv1')(x)
    x = BatchNormalization(name='conv1_BN')(x)

    # conv2
    x = ZeroPadding2D(padding= (1, 1))(x)
    x = MaxPooling2D(pool_size = (2,2), strides = 2, name='conv2_maxpool')(x)
    
    for i in range(3):
      x = residual_block(x, 256, 'conv2')

    # conv3
    for i in range(4):
      x = residual_block(x, 512, 'conv3')

    # conv4
    for i in range(6):
      x = residual_block(x, 1024, 'conv4')

    # conv5
    for i in range(3):
      x = residual_block(x, 2048, 'conv5')

    # last
    x = AveragePooling2D(padding = 'same', name='AvgPool')(x) # avg pooling 하고
    x = Flatten(name='flatten')(x)                            # flatten 하고
    x = Dense(self._num_classes, activation='softmax', name='fc')(x)        # Fully connected

    return x

# 예측하려는 class의 개수를 받음
class MyResNet50(Model):
  def __init__(self, num_classes):
    super(MyResNet50, self).__init__()
    self.num_cnn_list=[3,4,6,3]
    self.channel_list=[256, 512, 1024, 2048]
    self._num_classes = num_classes


  def call(self, input_data):
    x = ZeroPadding2D(padding= (3, 3))(input_data)
    
    # conv1
    x = Conv2D(filters=64, kernel_size = (7,7), strides = 2, name='conv1')(x)
    x = BatchNormalization(name='conv1_BN')(x)

    # conv2
    x = ZeroPadding2D(padding= (1, 1))(x)
    x = MaxPooling2D(pool_size = (2,2), strides = 2, name='conv2_maxpool')(x)
    
    for i in range(3):
      x = residual_block(x, 256, 'conv2')

    # conv3
    for i in range(4):
      x = residual_block(x, 512, 'conv3')

    # conv4
    for i in range(6):
      x = residual_block(x, 1024, 'conv4')

    # conv5
    for i in range(3):
      x = residual_block(x, 2048, 'conv5')

    # last
    x = AveragePooling2D(padding = 'same', name='AvgPool')(x) # avg pooling 하고
    x = Flatten(name='flatten')(x)                            # flatten 하고
    x = Dense(self._num_classes, activation='softmax', name='fc')(x)        # Fully connected

    return x

from tensorflow.keras.preprocessing.image import ImageDataGenerator
data_dir = './101_ObjectCategories'

# 이미지 데이터를 위한 이미지 제너레이터 생성
image_generator = ImageDataGenerator(rescale=1./255,      # 픽셀 값을 0~1사이로 조정
                                     validation_split=0.2 # 검증 데이터셋을 자동으로 준비
                                     )

# 학습 데이터셋 및 검증 데이터셋 로드

# flow_from_directory 메서드를 사용하여 데이터셋을 배치로 생성하고 클래스 레이블을 원-핫 인코딩된 벡터 형태로 반환
# flow_from_directory()는 인자로 전달해줌 directory의 하위 디렉토리 이름들을 레이블이라 간주.
# 레이블이라고 간주한 디렉토리 아래의 파일들을 해당 레이블의 이미지들이라고 알아서 추측하여 numpy array iterator 생성
train_generator = image_generator.flow_from_directory(
    data_dir,
    target_size=(224, 224),# 추후에 설계할 모델에 들어갈 input 이미지 사이즈를 (224, 224) 로 설정
    batch_size=256,       # 배치 사이즈
    class_mode='categorical', # multi-label 인데 one-hot, 'spars'로하면 멀티-레이블인데 레이블 인코딩
    subset='training'
)

validation_generator = image_generator.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model = MyResNet50(train_generator.num_classes)

# optimzier, loss_function 등을 설정해준다.
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics=['accuracy'])

# 모델을 실제 학습한다.
model.fit(train_generator, epochs=10, validation_data=validation_generator)

from tensorflow.keras.layers import AveragePooling2D, Flatten
from tensorflow.keras.applications import ResNet50

# 사전 훈련된 ResNet50 모델 불러오기 (include_top=False: Fully Connected Layer 제외)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 새로운 Fully Connected Layer 추가
x = base_model.output
x = AveragePooling2D()(x)  # Average Pooling Layer
x = Flatten()(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # Caltech-101의 클래스 수에 맞는 Output Layer

# 전체 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# 기존 ResNet50 모델의 가중치는 동결 (학습되지 않도록 설정)
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(train_generator, epochs=10, validation_data=validation_generator)

