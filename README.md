# handwrittenNumberRecognition
配置：
		python -V : 3.5.3
		tensorboard -V : 1.12.1
		tensorflow-gpu  -V : 1.12.0
		GPU : nvidia Geforce GTX 1050
		CUDA - V : 9.0
大家刚开始学习**tensorflow**的时候，最开始接触到的实战应该就是**手写数字识别**，因为tensorflow的中文官网上也以这个为例子，利用**MINST数据集**实现最简单的手写数字识别，网上有很多版本的代码都很杂乱，可能让刚开始学习的同学失去信心，下面将为大家讲解最简的数字识别。
首先这是我的整个工程的**目录**。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190104215312826.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xlZUdlNjY2,size_16,color_FFFFFF,t_70)

## 一、训练和测试（train&&test）
**1、获取数据集**
MNIST是在机器学习领域中的一个经典问题。该问题解决的是把**28x28**像素的灰度手写数字图片识别为相应的数字，其中数字的范围从0到9。

首先则是获得**训练**和**测试**的数据集，中文官网也给出了讲解
getData.py
```python
import input_data # 调用input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print('type of "mnist" is %s' % (type(mnist)))#获取训练数据的类型
print('number of train data is %d' % (mnist.train.num_examples))#输出训练数据多少个
print('number of test data is %d' % (mnist.test.num_examples))#输出测试数据有多少个
```
我们把下载后的数据集放在MNIST_data中
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190104220148687.png)
这是每个具体文件是什么，大家不用解压，就放在这个文件夹中，在训练的时候会自动解压读取数据。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190104215955672.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xlZUdlNjY2,size_16,color_FFFFFF,t_70)

**2、利用获得的数据进行训练和测试准确率**
这是用来训练和测试的python文件
这里选择的图片是28*28的图片，当然在完全理解之后也可以自己单独改成其他大小的图片。
---------      mnistdeep.py
```python
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])#设置图片大小28*28=784px
y_ = tf.placeholder(tf.float32, [None, 10])


def weight_variable(shape):#权重初始化
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):#偏置
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):#卷积
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):#池化
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])#权重张量形状[5, 5, 1, 32]
b_conv1 = bias_variable([32])#对应偏置量为[32]

x_image = tf.reshape(x,[-1, 28, 28, 1])#将x转化为一个4维向量，28*28表示宽高，1表示颜色通道，图片无颜色，若是有颜色的图片，则为3

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)#把x_image和权值张量进行卷积，加上偏置项，进行relu激活函数
h_pool1 = max_pool_2x2(h_conv1)                        #池化
#进行第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])#把类似的层叠起来，然后每个5*5的patch会得到64个特征
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#重复卷积和池化操作
h_pool2 = max_pool_2x2(h_conv2)

#密集卷积层
W_fc1 = weight_variable([7 * 7 * 64, 1024])#缩小图片尺寸，7*7加入一个1024尺寸的全连接层
b_fc1 = bias_variable([1024])               #初始化一个偏置量

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])#将池化层输出的张量转化为一个向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)#表示输出在神经元输出在dropout中保持不变，还可以自动处理神经元输出的scale

#输出层
W_fc2 = weight_variable([1024, 10])#权重
b_fc2 = bias_variable([10])#偏置

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)#添加一个softmax层

#训练和评估模型
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver() #定义saver

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):#20000次训练
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:#每100次迭代输出一次
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})#在feed_dict中添加keep_dict的比例
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    saver.save(sess, 'SAVE/model.ckpt') #模型储存位置

    print('test accuracy %g' % accuracy.eval(feed_dict={#最后得出测试最后的准确率
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

```
最后得出的正确率也比较感人，可以达到 **99.1%**
大家可以发现，最开始的正确率比较小，但是之后通过越来越多的训练次数，正确率逐渐趋近于1，最后得出的是平均的正确率。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190104221853772.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xlZUdlNjY2,size_16,color_FFFFFF,t_70)
我们把得到的训练模型存储**model.ckpt**到**SAVE**文件夹中
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190104221939278.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xlZUdlNjY2,size_16,color_FFFFFF,t_70)
得到的四个文件具体什么意思我就不具体阐述了，大家可以自行百度。

## 二、识别自己手写的数字图片
设置图片本身像素为**28*28**，可以在win10自带画图或则PS中自己用画笔写几个数字，保存为图片，之后例如我的test.jpg， test2.jpgs，test2.jpg
故意写了个不怎么像的6
![在这里插入图片描述](https://img-blog.csdnimg.cn/201901042225361.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xlZUdlNjY2,size_16,color_FFFFFF,t_70)
之后，测试识别的数字具体为多少
         ---------    testImg.py
```python
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt
import cutPic

def imageprepare(img):
    img = Image.open(img) #读取图片，路劲为相对路径，注意是28*28像素
    plt.imshow(img)  #显示需要识别的图片
    plt.show()
    img = img.convert('L')    #RGB转成灰色
    tv = list(img.getdata())#返回img的像素序列
    tva = [(255-x)*1.0/255.0 for x in tv]#得到每一个像素点的灰度，最大1为黑，最小0为白
    return tva


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')


def convolute_pool(img):
    result = imageprepare(img)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    # 第一层卷积
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 密集卷积层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 输出
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # 添加softmax层

    # 模型评估
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()  # 定义saver

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'SAVE/model.ckpt')

        prediction = tf.argmax(y_conv, 1)  # 返回对于y_conv预测到的标签值，与真实标签相比较比较是否匹配
        predint = prediction.eval(feed_dict={x: [result], keep_prob: 1.0}, session=sess)  # 得出测试的结果
		print('识别的数字为：')
        print(predint[0])


if __name__ == '__main__':
        img_file = 'test4.jpg'
        convolute_pool(img_file)
```
最前的一串代码和前一个有些相似，其实就是**卷积池化的过程**。
识别的效果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190104223115557.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xlZUdlNjY2,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190104223145280.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xlZUdlNjY2,size_16,color_FFFFFF,t_70)
我试了很多张，都是正确的，看来识别率还是很高的。

## 三、多个数字（一张图片）识别
首先我先自己随便画了张图，差多每个间隔**28像素**左右，因为切割的时候不会正好把某个数字切成一半，识别出来就不清楚了。保存为**testPic.jpg**
![在这里插入图片描述](https://img-blog.csdnimg.cn/201901042235163.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xlZUdlNjY2,size_16,color_FFFFFF,t_70)
**下面写切割图片的代码**
                    ------------------           cutPic.py
                   
```python
from PIL import Image

def cut(img_file, dx, dy):
    img = Image.open(img_file)
    n = 1
    x1 = 0
    y1 = 0
    x2 = dx
    y2 = dy
    while x2 <= img.size[1]:#纵向
        while y2 <= img.size[0]:#横向
            new_pic = 'pic/pic' + str(n) + '.jpg'
            #print('n=', n, ' x1=', x1, ' y1=', y1, ' x2=', x2, ' y2=', y2)
            img2 = img.crop((y1, x1, y2, x2))
            img2.save(new_pic)
            y1 += dy
            y2 = y1 + dy
            n += 1

        x1 = x1 + dx
        x2 = x1 + dx
        y1 = 0
        y2 = dy
        #print('------------------------------------------------------------')
    return n - 1


def main(img):
    n = cut(img, 28, 28)
    return n


if __name__ == '__main__':
    img = 'testPic.jpg'
    main(img)
```
我们把这些图片就保存在**pic文件夹**中，来看切割好的图片。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190104224418806.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xlZUdlNjY2,size_16,color_FFFFFF,t_70)
 已经**等距切割**好了，就差识别了，再次利用执行刚刚的数字识别py文件。

之后在**testImg()**中的主方法中利用循环逐个读取图片就行了。

```python
if __name__ == '__main__':
    img = 'testPic.jpg'
    n = cutPic.main(img)
    for i in range(1, 5):
        img_file = 'pic/pic' + str(i) + '.jpg'
        #print(img_file)
        convolute_pool(img_file)
```
 大家会发现，可以识别出第一个数字三，但之后的数字为什么不能识别出来了，出现的错误如下


```python
C:\Users\22833\AppData\Local\Programs\Python\Python35\python.exe "D:/python Projects/SophomoreProjects/tensorflowProjects/handwrittenNumberRecognition/testImg.py"
2019-01-04 22:47:21.088248: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-01-04 22:47:22.070566: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties: 
name: GeForce GTX 1050 major: 6 minor: 1 memoryClockRate(GHz): 1.493
pciBusID: 0000:01:00.0
totalMemory: 4.00GiB freeMemory: 3.30GiB
2019-01-04 22:47:22.070975: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-01-04 22:47:23.044184: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-04 22:47:23.044349: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2019-01-04 22:47:23.044450: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2019-01-04 22:47:23.044704: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3013 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1)
3
2019-01-04 22:47:25.138099: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-01-04 22:47:25.138278: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-04 22:47:25.138430: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2019-01-04 22:47:25.138594: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2019-01-04 22:47:25.138743: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3013 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1)
2019-01-04 22:47:25.747988: W tensorflow/core/framework/op_kernel.cc:1273] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable_10 not found in checkpoint
Traceback (most recent call last):
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\client\session.py", line 1334, in _do_call
    return fn(*args)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\client\session.py", line 1319, in _run_fn
    options, feed_dict, fetch_list, target_list, run_metadata)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\client\session.py", line 1407, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.NotFoundError: Key Variable_10 not found in checkpoint
	 [[{{node save_1/RestoreV2}} = RestoreV2[dtypes=[DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, ..., DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT], _device="/job:localhost/replica:0/task:0/device:CPU:0"](_arg_save_1/Const_0_0, save_1/RestoreV2/tensor_names, save_1/RestoreV2/shape_and_slices)]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 1546, in restore
    {self.saver_def.filename_tensor_name: save_path})
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\client\session.py", line 929, in run
    run_metadata_ptr)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\client\session.py", line 1152, in _run
    feed_dict_tensor, options, run_metadata)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\client\session.py", line 1328, in _do_run
    run_metadata)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\client\session.py", line 1348, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.NotFoundError: Key Variable_10 not found in checkpoint
	 [[node save_1/RestoreV2 (defined at D:/python Projects/SophomoreProjects/tensorflowProjects/handwrittenNumberRecognition/testImg.py:82)  = RestoreV2[dtypes=[DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, ..., DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT], _device="/job:localhost/replica:0/task:0/device:CPU:0"](_arg_save_1/Const_0_0, save_1/RestoreV2/tensor_names, save_1/RestoreV2/shape_and_slices)]]

Caused by op 'save_1/RestoreV2', defined at:
  File "D:/python Projects/SophomoreProjects/tensorflowProjects/handwrittenNumberRecognition/testImg.py", line 99, in <module>
    convolute_pool(img_file)
  File "D:/python Projects/SophomoreProjects/tensorflowProjects/handwrittenNumberRecognition/testImg.py", line 82, in convolute_pool
    saver = tf.train.Saver()  # 定义saver
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 1102, in __init__
    self.build()
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 1114, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 1151, in _build
    build_save=build_save, build_restore=build_restore)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 795, in _build_internal
    restore_sequentially, reshape)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 406, in _AddRestoreOps
    restore_sequentially)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 862, in bulk_restore
    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\ops\gen_io_ops.py", line 1550, in restore_v2
    shape_and_slices=shape_and_slices, dtypes=dtypes, name=name)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 787, in _apply_op_helper
    op_def=op_def)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\util\deprecation.py", line 488, in new_func
    return func(*args, **kwargs)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\framework\ops.py", line 3274, in create_op
    op_def=op_def)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\framework\ops.py", line 1770, in __init__
    self._traceback = tf_stack.extract_stack()

NotFoundError (see above for traceback): Key Variable_10 not found in checkpoint
	 [[node save_1/RestoreV2 (defined at D:/python Projects/SophomoreProjects/tensorflowProjects/handwrittenNumberRecognition/testImg.py:82)  = RestoreV2[dtypes=[DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, ..., DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT], _device="/job:localhost/replica:0/task:0/device:CPU:0"](_arg_save_1/Const_0_0, save_1/RestoreV2/tensor_names, save_1/RestoreV2/shape_and_slices)]]


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 1556, in restore
    names_to_keys = object_graph_key_mapping(save_path)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 1830, in object_graph_key_mapping
    checkpointable.OBJECT_GRAPH_PROTO_KEY)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 371, in get_tensor
    status)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\framework\errors_impl.py", line 528, in __exit__
    c_api.TF_GetCode(self.status.status))
tensorflow.python.framework.errors_impl.NotFoundError: Key _CHECKPOINTABLE_OBJECT_GRAPH not found in checkpoint

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:/python Projects/SophomoreProjects/tensorflowProjects/handwrittenNumberRecognition/testImg.py", line 99, in <module>
    convolute_pool(img_file)
  File "D:/python Projects/SophomoreProjects/tensorflowProjects/handwrittenNumberRecognition/testImg.py", line 86, in convolute_pool
    saver.restore(sess, 'SAVE/model.ckpt')
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 1562, in restore
    err, "a Variable name or other graph key that is missing")
tensorflow.python.framework.errors_impl.NotFoundError: Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key Variable_10 not found in checkpoint
	 [[node save_1/RestoreV2 (defined at D:/python Projects/SophomoreProjects/tensorflowProjects/handwrittenNumberRecognition/testImg.py:82)  = RestoreV2[dtypes=[DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, ..., DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT], _device="/job:localhost/replica:0/task:0/device:CPU:0"](_arg_save_1/Const_0_0, save_1/RestoreV2/tensor_names, save_1/RestoreV2/shape_and_slices)]]

Caused by op 'save_1/RestoreV2', defined at:
  File "D:/python Projects/SophomoreProjects/tensorflowProjects/handwrittenNumberRecognition/testImg.py", line 99, in <module>
    convolute_pool(img_file)
  File "D:/python Projects/SophomoreProjects/tensorflowProjects/handwrittenNumberRecognition/testImg.py", line 82, in convolute_pool
    saver = tf.train.Saver()  # 定义saver
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 1102, in __init__
    self.build()
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 1114, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 1151, in _build
    build_save=build_save, build_restore=build_restore)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 795, in _build_internal
    restore_sequentially, reshape)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 406, in _AddRestoreOps
    restore_sequentially)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\training\saver.py", line 862, in bulk_restore
    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\ops\gen_io_ops.py", line 1550, in restore_v2
    shape_and_slices=shape_and_slices, dtypes=dtypes, name=name)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 787, in _apply_op_helper
    op_def=op_def)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\util\deprecation.py", line 488, in new_func
    return func(*args, **kwargs)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\framework\ops.py", line 3274, in create_op
    op_def=op_def)
  File "C:\Users\22833\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\framework\ops.py", line 1770, in __init__
    self._traceback = tf_stack.extract_stack()

NotFoundError (see above for traceback): Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key Variable_10 not found in checkpoint
	 [[node save_1/RestoreV2 (defined at D:/python Projects/SophomoreProjects/tensorflowProjects/handwrittenNumberRecognition/testImg.py:82)  = RestoreV2[dtypes=[DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, ..., DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT], _device="/job:localhost/replica:0/task:0/device:CPU:0"](_arg_save_1/Const_0_0, save_1/RestoreV2/tensor_names, save_1/RestoreV2/shape_and_slices)]]


Process finished with exit code 1

```
我也是查了很久才找到解决方法，错误的意思就是有些节点已经存在，不能再次新建，方法就是**重置**，清理已经**存在的节点和计算图**。
在方法第一行加一个重置就行了。

```python
def convolute_pool(img):
    tf.reset_default_graph()
```


发现会输出一些其他关于tensoflow以及显卡的一些信息，我们把这些信息忽略掉，在testImg()中加入即可解决。

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```
输出结果，成功识别出图片中的数字。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190104225904131.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190104225647804.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xlZUdlNjY2,size_16,color_FFFFFF,t_70)
大家如果有需要请留言邮箱，我会把整个工程的**源码**发给大家学习。

参考资料：http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html
