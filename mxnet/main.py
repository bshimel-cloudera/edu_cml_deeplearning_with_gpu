#
# Main PyTorch Code Loop
#

########
########  Setup Environment and install libs
########

####
#### Installing Required Libraries
####

%pip install mxnet-cu110==1.9.0b20210226 -f https://dist.mxnet.io/python/cu110
%pip install scikit-learn pandas

# Load Libraries
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix



# Check if CUDA is loaded properly
mx.context.num_gpus()
ctx = mx.gpu()

########
########  Build and train model
########

####
#### Setup the Dataset
####


# define the transformations
# in Gluon we transform first
def transformations(data, label):
    data = mx.nd.transpose(data, (2,0,1))
    data = data.astype(np.float32)
    label = label.astype(np.float32)
    return data, label

# Load data from the Gluon default data libraries
train = mx.gluon.data.vision.FashionMNIST(train=True, transform=transformations)
test = mx.gluon.data.vision.FashionMNIST(train=False, transform=transformations)

# Build the DataLoaders
batch_size = 100

train_data = mx.gluon.data.DataLoader(
    train,
    batch_size=batch_size, shuffle=True, last_batch='discard')

test_data = mx.gluon.data.DataLoader(
    test,
    batch_size=batch_size, shuffle=False, last_batch='discard')

####
#### Setup the Model and training task
####

# Instantiate the Model
simple_network = mx.gluon.nn.Sequential()
with simple_network.name_scope():
    simple_network.add(mx.gluon.nn.Conv2D(channels=32, kernel_size=3, activation='relu'))
    simple_network.add(mx.gluon.nn.BatchNorm(momentum=0.1))
    simple_network.add(mx.gluon.nn.MaxPool2D(pool_size=2, strides=2))
    simple_network.add(mx.gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
    simple_network.add(mx.gluon.nn.BatchNorm(momentum=0.1))
    simple_network.add(mx.gluon.nn.MaxPool2D(pool_size=2))
    simple_network.add(mx.gluon.nn.Dense(600))
    simple_network.add(mx.gluon.nn.Dropout(rate=0.5))
    simple_network.add(mx.gluon.nn.Dense(120))
    simple_network.add(mx.gluon.nn.Dense(10))

# Initialize Parameters
simple_network.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

# optimisers
trainer = mx.gluon.Trainer(simple_network.collect_params(), 'adam', {'learning_rate': .001})

# Loss Function
softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()

# Setting the evaluation function
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


####
#### Train the Model
####

epochs = 5
smoothing_constant = .01
count = 0

# collecting loss and accuracy for visualising
iteration_list = []
loss_list = []

epoch_list = []
accuracy_list = []

for e in range(epochs):
    for i, (d, l) in enumerate(train_data):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        with autograd.record():
            output = simple_network(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])

        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
        
        iteration_list.append(count)
        loss_list.append(curr_loss)


        count +=1

    test_accuracy = evaluate_accuracy(test_data, simple_network)
    train_accuracy = evaluate_accuracy(train_data, simple_network)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

    epoch_list.append(e)
    accuracy_list.append(train_accuracy)

####
#### Look at model train result
####

#### Visualise the training Loss
plt.plot(iteration_list, loss_list)

#### Visualise the training accuracy
plt.plot(epoch_list, accuracy_list)

### Confusion Matrix Report
labels_list = []
output_list = []

for i, batch in enumerate(test_data):
    data = mx.gluon.utils.split_and_load(batch[0], ctx_list=[ctx], batch_axis=0)
    label = mx.gluon.utils.split_and_load(batch[1], ctx_list=[ctx], batch_axis=0)
    outputs = [simple_network(x) for x in data]
    
    for l in label:
        labels_list.extend(l.asnumpy().tolist())
    for o in outputs[0]:
        output_list.append(np.argmax(o.asnumpy()))    

confusion_matrix(labels_list, output_list)
print("Classification report for CNN :\n%s\n"
      % (metrics.classification_report(labels_list, output_list)))


