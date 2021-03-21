keep_prob=tf.placeholder(tf.float32,[])
'''
embedding
'''
sequence_length=tf.einsum('bw->b',tf.cast(x>0,tf.int32))
embedding_table=tf.get_variable('embedding_table',[3,128])###
embedding_table=tf.concat([embedding_table,np.load('skip_gram_word_embedding.npy')[3:]],0)###
E=tf.nn.embedding_lookup(embedding_table,x)
E=tf.nn.dropout(E,keep_prob)

'''
CNN
'''
E1=tf.layers.dense(E,64,tf.nn.relu)
E2=tf.layers.conv1d(E,64,2,padding='same',activation=tf.nn.relu,use_bias=True)
E3=tf.layers.conv1d(E,64,3,padding='same',activation=tf.nn.relu,use_bias=True)
region_embedding=tf.concat([E1,E2,E3],axis=-1)
region_embedding=norm(region_embedding)
region_embedding=tf.nn.dropout(region_embedding,keep_prob)
'''
GRU
'''
sequence_length=tf.einsum('bw->b',tf.cast(x>0,tf.int32))
fw_cell = tf.nn.rnn_cell.GRUCell(64,tf.nn.tanh)
bw_cell = tf.nn.rnn_cell.GRUCell(64,tf.nn.tanh)
outputs,state = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                bw_cell,
                                                region_embedding,
                                                sequence_length=sequence_length,
                                                dtype=tf.float32)

state = tf.concat(state, -1)
outputs=tf.concat(outputs,-1)
'''
attention
'''
Q=tf.layers.dense(state,128,tf.nn.relu)
K=tf.layers.dense(outputs,128,tf.nn.relu)
V=tf.layers.dense(outputs,128,tf.nn.relu)

Q=tf.nn.dropout(Q,keep_prob)
K=tf.nn.dropout(K,keep_prob)
V=tf.nn.dropout(V,keep_prob)

score=tf.nn.softmax(tf.einsum('bd,bwd->bw',Q,K))
score=tf.nn.dropout(score,keep_prob)

output=tf.einsum('bw,bwd->bd',score,V)

y=output
