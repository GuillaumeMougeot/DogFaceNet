# Training
next_images, next_labels = next_element

output = model(next_images)

logit = arcface_loss(embedding=output, labels=next_labels,
                     w_init=None, out_num=count_labels)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logit, labels=next_labels))

# Validation
next_images_valid, next_labels_valid = next_valid

output_valid = model(next_images_valid)

logit_valid = arcface_loss(embedding=output_valid, labels=next_labels_valid,
                     w_init=None, out_num=count_labels)
loss_valid = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logit_valid, labels=next_labels_valid))

pred_valid = tf.nn.softmax(logit_valid)
acc_valid = tf.reduce_mean(tf.cast(tf.equal(tf.argmin(pred_valid, axis=1), next_labels_valid), dtype=tf.float32))

# Optimizer
lr = 0.01

opt = tf.train.AdamOptimizer(learning_rate=lr)
train = opt.minimize(loss)

# Accuracy for validation and testing
pred = tf.nn.softmax(logit)
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmin(pred, axis=1), next_labels), dtype=tf.float32))


############################################################
#  Training session
############################################################


init = tf.global_variables_initializer()

with tf.Session() as sess:

    summary = tf.summary.FileWriter('../output/summary', sess.graph)
    summaries = []
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))
    summaries.append(tf.summary.scalar('inference_loss', loss))
    summary_op = tf.summary.merge(summaries)
    saver = tf.train.Saver(max_to_keep=100)

    sess.run(init)

    # Training
    nrof_batches = len(filenames_train)//BATCH_SIZE + 1
    nrof_batches_valid = len(filenames_train)//BATCH_SIZE + 1

    print("Start of training...")
    for i in range(EPOCHS):
        
        feed_dict = {filenames_train_placeholder: filenames_train,
                     labels_train_placeholder: labels_train}

        sess.run(iterator.initializer, feed_dict=feed_dict)

        feed_dict_valid = {filenames_valid_placeholder: filenames_valid,
                           labels_valid_placeholder: labels_valid}

        sess.run(it_valid.initializer, feed_dict=feed_dict_valid)

        # Training
        for j in trange(nrof_batches):
            try:
                _, loss_value, summary_op_value, acc_value = sess.run((train, loss, summary_op, acc))
                # summary.add_summary(summary_op_value, count)
                tqdm.write("\n Batch: " + str(j)
                    + ", Loss: " + str(loss_value)
                    + ", Accuracy: " + str(acc_value)
                    )

            except tf.errors.OutOfRangeError:
                break
        
        # Validation
        print("Start validation...")
        tot_acc = 0
        for _ in trange(nrof_batches_valid):
            try:
                loss_valid_value, acc_valid_value = sess.run((loss_valid, acc_valid))
                tot_acc += acc_valid_value
                tqdm.write("Loss: " + str(loss_valid_value)
                    + ", Accuracy: " + str(acc_valid_value)
                    )

            except tf.errors.OutOfRangeError:
                break
        print("End of validation. Total accuray: " + str(tot_acc/nrof_batches_valid))


    print("End of training.")
    print("Start evaluation...")
    # Evaluation on the validation set:
    ## One-shot training
    #sess.run()