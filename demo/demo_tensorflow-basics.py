import tensorflow as tf

##########################################
#
#
# Memo
# ----
#   1. Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
#      <ref> https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
#
#

input1 = tf.ones((2, 3))
input2 = tf.reshape(tf.range(1, 7, dtype=tf.float32), (2, 3))
output = input1 + input2

with tf.Session():
  result = output.eval()
print 'output>\n%s\n' % result  