import tensorflow as tf

num_classes = 10
img_rows, img_cols = 28, 28
num_channels = 1
input_shape = (img_rows, img_cols, num_channels)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.summary()


# here are some comments
# $ end of the line
# * find a word
# D delete rest of the line
# d delete the current line
# w move forward
# b move back
# v visual mode
# 0 move to start of the line
# % toggle brackets
# set number & set nonumber
# d t [any character] delete line till the character
# u undo
# ^r redo
# a amend
# i insert
# y copy
# p paste with enter
# P paste without enter
# c change line with insert mode
# r replace a char