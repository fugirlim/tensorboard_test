import tensorflow as tf
from datetime import datetime

# python
#tensorboard --logdir=logs/fit

# Define a function to parse the log document
def parse_log(log_file):
    learning_rates = []
    accuracies = []
    losses = []

    with open(log_file, 'r',encoding='utf-8') as f:
        for line in f:
            if "step" in line:
                loss = line.split(' ')[-1].strip()
            elif '率' in line:
                losses.append(float(loss))
                if '学习率' in line:
                    learning_rate = float(line.split('：')[1])
                    learning_rates.append(learning_rate)
                elif '准确率' in line:
                    accuracy = float(line.split(':')[-1].strip().strip('%'))
                    accuracies.append(accuracy)

    losses = losses[::2]
    if len(losses) > len(learning_rates):
        losses = losses[0: len(learning_rates)]
    elif len(losses) < len(learning_rates):
        learning_rates = learning_rates[0: len(losses)]
        accuracies = accuracies[0: len(losses)]
    epochs = list(range(len(losses)))

    return epochs, learning_rates, accuracies, losses


# Parse the log document
log_file = 'f6_logs_0606.log'
epochs, learning_rates, accuracies, losses = parse_log(log_file)

# Create a TensorFlow SummaryWriter to write summary objects to an event file
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#
summary_writer = tf.summary.create_file_writer(log_dir)

# Write learning rate and accuracy data as summary objects
with summary_writer.as_default():
    for step, (epoch, learning_rate, accuracy, loss) in enumerate(zip(epochs, learning_rates, accuracies, losses)):
        tf.summary.scalar('learning_rate', learning_rate, step=epoch)
        tf.summary.scalar('accuracy', accuracy, step=epoch)
        tf.summary.scalar('loss', loss, step=epoch)

# Close the summary writer
summary_writer.close()

# Start TensorBoard to visualize the logs
print("To visualize the logs, run the following command in your terminal:")
print(f"tensorboard --logdir={log_dir}")
