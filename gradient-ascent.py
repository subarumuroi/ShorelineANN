import tensorflow as tf
from tensorflow.keras.models import load_model

csvFile = "C:/Users/S2996310/Subaru/AllRuns_SingleTime_Train.csv"
testFile = "C:/Users/S2996310/Subaru/AllRuns_SingleTime_Test.csv"
midFile = "C:/Users/S2996310/Subaru/avg of mid points.csv"
testmidFile = "C:/Users/S2996310/Subaru/test avg of mid points.csv"

Adam_learning_rate = 0.0066
Nepochs = 2000
Nneurons_Dense = 320
activation = 'relu'
batch = 32

# Load the model
model = load_model('model')  

# Init random 6-dim input vector
input_vector = tf.Variable(tf.random.normal([1, 6]), trainable=True)

# Choose an optimiser
optimizer = tf.optimizers.Adam(learning_rate=Adam_learning_rate)

# Gradient ascent iterations
for i in range(100):  # Number of iterations
    with tf.GradientTape() as tape:
        tape.watch(input_vector)
        prediction = model(input_vector)  # Get model's prediction

    # Compute gradients of the prediction with respect to input vector
    gradients = tape.gradient(prediction, input_vector)
    
    # Apply gradients to input vector i.e. maximise pred
    optimizer.apply_gradients([(gradients, input_vector)])

    # Testing on constraining input vector in this code...in development at this stage
    # e.g., input_vector.assign(tf.clip_by_value(input_vector, min_value, max_value))

optimized_input = input_vector.numpy()
print("Optimized Input:", optimized_input)