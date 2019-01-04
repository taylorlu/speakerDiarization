import numpy as np
import os
import librosa
import tensorflow as tf
from functools import reduce

SRC_PATH = r'E:\AAA\train'
nfft = 512
window = 0.025
hop = 0.01
tisv_frame = 80
hidden = 768
proj = 256
num_layer = 3

def normalize(x):
    """ normalize the last dimension vector of the input matrix
    :return: normalized input
    """
    return x/tf.sqrt(tf.reduce_sum(x**2, axis=-1, keep_dims=True)+1e-6)

def initDVectorModel():

    batch = tf.placeholder(shape=[None, None, 40], dtype=tf.float32) # enrollment batch (time x batch x n_mel)
    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=hidden, num_proj=proj) for i in range(num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # make lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize

    return batch, embedded

def similar(matrix):  # calc d-vectors similarity in pretty format output.
    ids = matrix.shape[0]
    for i in range(ids):
        for j in range(ids):
            dist = matrix[i,:]*matrix[j,:]
            dist = reduce(lambda x, y: x+y, dist)
            # dist = np.linalg.norm(matrix[i,:] - matrix[j,:])
            print('%.2f  ' % dist, end='')
            if((j+1)%3==0 and j!=0):
                print("| ", end='')
        if((i+1)%3==0 and i!=0):
            print('\n')
            print("*"*80, end='')
        print("\n")

def main():
    """The main function."""

    train_cluster_id = []
    train_sequence = None
    ccc = 0
    batch, embedded = initDVectorModel()  # d-vector pre-trained model, by 'GE2E' paper.

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.restore(sess, r'C:\Users\Administrator\Desktop\pictures\tisv_model\Check_Point\model.ckpt-1')

        files = os.listdir(SRC_PATH)
        for f in files:         # Deal every wav file, VAD and Concat, slide by overlay 50%, extract d-vectors in batch.

            spk = f.split('_')[0]
            utter = f.split('_')[1].split('.')[0]
            label = '{}_{}'.format(spk, utter)
            print("label = {}".format(label))
            
            utter_path = os.path.join(SRC_PATH, f)

            utter, sr = librosa.core.load(utter_path, sr=16000)        # load utterance audio
            intervals = librosa.effects.split(utter, top_db=20)         # voice activity detection

            summ = np.array([])
            for interval in intervals:
              utter_part = utter[interval[0]:interval[1]]
              summ = np.concatenate([summ, utter_part])

            cur_slide = 0
            mfcc_win_sample = int(sr*hop*tisv_frame)
            utterances_spec = []

            while(True):  # slide window.
                if(cur_slide + mfcc_win_sample > summ.shape[0]):
                    break
                slide_win = summ[cur_slide : cur_slide+mfcc_win_sample]

                S = librosa.feature.mfcc(y=slide_win, sr=sr, n_mfcc=40)
                input = S.transpose((1,0))
                utterances_spec.append(input)

                cur_slide += int(mfcc_win_sample/2)

            utterances_spec = np.array(utterances_spec)
            utterances_spec = utterances_spec.transpose((1,0,2))

            vectors = sess.run(embedded, feed_dict={batch: utterances_spec})  # d-vectors of this wav file.
            train_cluster_id.extend([label]*vectors.shape[0])

            if(train_sequence is None):
                train_sequence = vectors
            else:
                train_sequence = np.vstack((train_sequence, vectors))

    train_cluster_id = np.array(train_cluster_id)

    print('train_sequence = {}'.format(train_sequence.shape))
    print('train_cluster_id = {}'.format(train_cluster_id.shape))

    np.savez('training_data', train_sequence=train_sequence, train_cluster_id=train_cluster_id)

if __name__ == '__main__':
  main()
