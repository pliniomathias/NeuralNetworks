
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Definir Parâmetros
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

# Inputs do TensorFlow
x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

# Criar um modelo

# Definir Pesos
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
    # Construir um modelo linear
    model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
    

w_h = tf.histogram_summary("Pesos", W)
b_h = tf.histogram_summary("biases", b)


with tf.name_scope("cost_function") as scope:
    # Minimizar o erro usando entropia cruzada
    cost_function = -tf.reduce_sum(y*tf.log(model))
    # Criar um sumário para monitorar o custo da função
    tf.scalar_summary("cost_function", cost_function)

with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Inicializando as variáveis
init = tf.initialize_all_variables()

# Juntar todos os sumários em um operador único
merged_summary_op = tf.merge_all_summaries()

with tf.Session() as sess:
    sess.run(init)

    
    
    # Mudar de acordo com a localização no computador
    summary_writer = tf.train.SummaryWriter('/LOCATION/ON/YOUR/COMPUTER/', graph_def=sess.graph_def)

    # ciclo de treino
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
     
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
  
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Computar a média das perdas
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            # Escrever logs para cada iteração
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + i)
        # Mostrar logs por passos de cada iteração
        if iteration % display_step == 0:
            print "Iteracao:", '%04d' % (iteration + 1), "custo=", "{:.9f}".format(avg_cost)

    print "Ajuste completo!"

    # Testar o modelo
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # Calcular a precisão
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
print "Precisao:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})