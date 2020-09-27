import tensorflow as tf


def run():
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

    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, verbose=1,
              validation_data=(x_test, y_test))

    model.summary()


if __name__ == "__main__":
    run()

'''
 here are some comments
 x cut
 y copy
 p, P paste
 $ end of the line
 * find a word
 d delete rest of the line
 d delete the current line
 w move forward
 b move back
 v visual mode
 0 move to start of the line
 % toggle brackets
 set number & set nonumber
 d t [any character] delete line till the character
 u undo
 ^r redo
 a amend, a amend to end of the line
 i insert
 y copy
 p paste with enter
 p paste without enter
 c change line with insert mode, c change rest of the line
 r replace a char
 t [char] or f [char]
 ; next
 x delete
 cw : change word
 dw : delete word
 yw : copy word
 z : center the cursor
 <> : indent
 q : macro
 ^v + ^i -> macro insert
'''

'''
q practice
q to w -> record
@W -> execute 
const arr = [
    ''One''
    ''Two''
    ''three''
    ''Four''
    ''Five''
    ''Six''
    ''Seven''
    ''Eight''
    ''Nine''
    ''Ten''
] 
'''
