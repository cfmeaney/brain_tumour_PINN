################################################################################
### PINN for Proliferation-Invasion Model of Glioma in 2D (Discrete Data)
################################################################################

# Get rid of warning messages and CUDA loadings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# Import Packages
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.animation as animation
from matplotlib import cm
import tensorflow_probability as tfp
import time

# Start Timing Code
t0 = time.time()

# Accessible Definitions
path = 'Data/Fisher_2D_2Eq_Nonsymmetric.mat' # Location of FEniCS data
D = 0.01 # Diffusivity in PDE
r = 0.5 # Proliferation rate in PDE
D_guess = 0.01 # intial guess for diffusivity in PDE
r_guess = 0.01 # intial guess for proliferation rate in PDE
Adam_Epochs = 20000 # Number of epochs of Adam optimization
q = 500 # Number of RK time steps (max 500)
noise = 0.01 # number of standard deviations of output noise

# Load in RK weights
IRK_weights = np.float32(np.loadtxt('IRK_weights/Butcher_IRK' + str(q) + '.txt', ndmin=2))
IRK_times = IRK_weights[q**2+q:]
IRK_weights = IRK_weights[:q**2+q].reshape((q+1,q))
IRK_alpha = tf.constant(IRK_weights[:-1,:], dtype='float32')
IRK_beta = tf.constant(IRK_weights[-1:,:], dtype='float32')

# On/offs for code grouping
load_data = 'on'
data_preprocessing = 'on'
define_model = 'on'
train_model = 'on'
plotting = 'on'
printing = 'on'
save_ani = 'off' # Plotting needs to also be on for this to work

if load_data == 'on':
    # load in FEniCS data
    data = scipy.io.loadmat(path)
    # Extract the time, positions, and solution
    t = data['t'].flatten()[:,None]
    x = data['x'][:,0].flatten()[:,None]
    y = data['x'][:,1].flatten()[:,None]
    Exact_u = np.abs(data['u1'])
    # Create a time/position grid
    X, T = np.meshgrid(x,t)
    Y, T = np.meshgrid(y,t)
    # Flatten the position/time and solution arrays
    XT_grid = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    YT_grid = np.hstack((Y.flatten()[:,None], T.flatten()[:,None]))
    xyt = np.array([XT_grid[:,0], YT_grid[:,0], XT_grid[:,1]]).T # all (x,y,t) combos in an array (shape = (x*y*t, 3))
    u_star = np.array([])
    for i in range(len(t)):
        u_star = np.append(u_star, Exact_u[:,:,i].T.flatten())
    u_star = u_star.reshape(len(u_star),1) # values of the solution corresponding to the inputs of xyt

if data_preprocessing == 'on':
    # Get spatial input
    train_inputs = xyt[:int(np.sqrt(x.shape[0]))*int(np.sqrt(y.shape[0])),0:2]
    # Create snapshot outputs
    train_outputs_t0 = u_star[:int(np.sqrt(x.shape[0]))*int(np.sqrt(y.shape[0])),0] # Initial snapshot solution data
    train_outputs_t1 = u_star[-int(np.sqrt(x.shape[0]))*int(np.sqrt(y.shape[0])):,0] # Final snapshot solution data
    train_outputs_t0_noise = train_outputs_t0 + noise*np.std(train_outputs_t0)*np.random.randn(train_outputs_t0.shape[0]) # Initial snapshot solution data with noise
    train_outputs_t1_noise = train_outputs_t1 + noise*np.std(train_outputs_t1)*np.random.randn(train_outputs_t1.shape[0]) # Final snapshot solution data with noise
    # Define RK step size
    dt = (t[-1] - t[0])[0]
    # Normalize train_inputs
    input_min = train_inputs.min(0)
    input_max = train_inputs.max(0)
    train_inputs_norm = 2.0*(train_inputs - input_min)/(input_max - input_min) - 1.0

if define_model == 'on':

    # Define model architecture
    Inputs = tf.keras.layers.Input(shape=(2,))
    Dense_1 = tf.keras.layers.Dense(50, activation='tanh', kernel_initializer="glorot_normal")(Inputs)
    Dense_2 = tf.keras.layers.Dense(50, activation='tanh', kernel_initializer="glorot_normal")(Dense_1)
    Dense_3 = tf.keras.layers.Dense(50, activation='tanh', kernel_initializer="glorot_normal")(Dense_2)
    Dense_4 = tf.keras.layers.Dense(50, activation='tanh', kernel_initializer="glorot_normal")(Dense_3)
    Prediction = tf.keras.layers.Dense(q)(Dense_4)

    # Define layers of prediction network
    model_prediction = tf.keras.models.Model(inputs=Inputs, outputs=Prediction)

    # Define TF variables for autodifferentiation
    train_inputs_norm = train_inputs_norm.astype(np.float32)
    train_inputs_var = tf.Variable(train_inputs_norm, name='train_inputs_var')
    dummy = tf.ones([train_inputs_var.shape[0], q], dtype=np.float32)

    # Create new layer that implements loss function
    class Loss_Layer(tf.keras.layers.Layer):
        def __init__(self, D_var, r_var):
            super(Loss_Layer, self).__init__()
            self.D_var = tf.Variable(np.array([D_var]))
            self.r_var = tf.Variable(np.array([r_var]))

        # Define custom loss function
        def custom_loss(self, pred):
            # Setep tape and perform derivatives
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(train_inputs_var)
                tape.watch(dummy)
                k = model_prediction(train_inputs_var)
                g_U = tape.gradient(k, train_inputs_var, output_gradients=dummy)
                g_Ux = g_U[:,0]
                g_Uy = g_U[:,1]
                k_x = tape.gradient(g_Ux, dummy)
                k_y = tape.gradient(g_Uy, dummy)
                g_Ux_U = tape.gradient(k_x, train_inputs_var, output_gradients=dummy)
                g_Uy_U = tape.gradient(k_y, train_inputs_var, output_gradients=dummy)
                g_Ux_Ux = g_Ux_U[:,0]
                g_Uy_Uy = g_Uy_U[:,1]
            k_xx = tape.gradient(g_Ux_Ux, dummy)
            k_yy = tape.gradient(g_Uy_Uy, dummy)

            # Define scale factors and scale derivatives
            scale_x = 2/(input_max[0]-input_min[0])
            scale_y = 2/(input_max[1]-input_min[1])
            k_x = k_x*scale_x
            k_y = k_y*scale_y
            k_xx = k_xx*scale_x**2
            k_yy = k_yy*scale_y**2

            # Implement PDE
            D = tf.exp(self.D_var)
            r = tf.exp(self.r_var)
            u = k
            u_x = k_x
            u_y = k_y
            u_xx = k_xx
            u_yy = k_yy
            u_t = D*(u_xx + u_yy) + r*u*(1.0-u)

            # Apply IRK Scheme to find initial and final snapshots
            U0_pred = k - dt*tf.matmul(u_t, tf.transpose(IRK_alpha))
            U1_pred = k + dt*tf.matmul(u_t, tf.transpose(IRK_beta - IRK_alpha))

            # Format exact solutions for loss
            U0 = q*[train_outputs_t0_noise]
            U0 = tf.stack(U0, axis=1)
            U0 = tf.cast(U0, tf.float32)
            U1 = q*[train_outputs_t1_noise]
            U1 = tf.stack(U1, axis=1)
            U1 = tf.cast(U1, tf.float32)

            # Calculate loss
            MSE = tf.reduce_mean(tf.square(U0 - U0_pred)) + tf.reduce_mean(tf.square(U1 - U1_pred))

            return MSE

        def call(self, pred):
            self.add_loss(self.custom_loss(pred))
            return pred

    # Add Loss_Layer to model
    D_guess = np.log(D_guess)
    r_guess = np.log(r_guess)
    D_guess = D_guess.astype(np.float32)
    r_guess = r_guess.astype(np.float32)
    My_Loss = Loss_Layer(D_guess, r_guess)(Prediction)

    # Create trainable model with custom loss function
    model_loss = tf.keras.models.Model(inputs=Inputs, outputs=My_Loss)

if train_model == 'on':

    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, loss, logs=None):
            D = np.exp(model_loss.layers[-1].get_weights()[0])
            r = np.exp(model_loss.layers[-1].get_weights()[1])
            loss = loss['loss']
            print('Epoch:', epoch+1, '... Loss:', round(loss, 10), '... D value:', round(D[0], 5), '... r value:', round(r[0], 5))

    # Compile model for Adam optimization
    model_loss.compile(optimizer=tf.keras.optimizers.Adam())
    # Execute Adam optimization
    history = model_loss.fit(train_inputs_norm, None, epochs=Adam_Epochs, batch_size=train_inputs_norm.shape[0], verbose=0, callbacks=[CustomCallback()])
    # Get elapsed time
    t1 = time.time()
    Total_Time = t1-t0

if plotting == 'on':
    # Create predicted solution
    k_pred = model_prediction.predict(train_inputs_norm)
    k_pred = k_pred.reshape((int(np.sqrt(x.shape[0])), int(np.sqrt(y.shape[0])), q))
    train_outputs_t0 = train_outputs_t0.reshape((int(np.sqrt(x.shape[0])), int(np.sqrt(y.shape[0]))))
    train_outputs_t1 = train_outputs_t1.reshape((int(np.sqrt(x.shape[0])), int(np.sqrt(y.shape[0]))))
    U_pred = np.dstack((train_outputs_t0, k_pred, train_outputs_t1))

    # Create predicted and exact solutions with same time stepping
    uniform_times = t
    IRK_times = IRK_times*(t[-1]-t[0])+t[0]
    num_plot_times = 2*q
    dt_plot = dt/num_plot_times
    t_plot = np.linspace(uniform_times[0], uniform_times[-1], num_plot_times)
    Exact_plot = np.zeros((Exact_u.shape[0], Exact_u.shape[1], num_plot_times))
    U_pred_plot = np.zeros((U_pred.shape[0], U_pred.shape[1], num_plot_times))
    for t in range(num_plot_times-1):
        i = 0
        while uniform_times[i] < t_plot[t][0]:
            i = i + 1
        j = 0
        while IRK_times[j] < t_plot[t][0]:
            j = j + 1
        Exact_plot[:,:,t] = Exact_u[:,:,i]
        U_pred_plot[:,:,t] = U_pred[:,:,j]
    Exact_plot[:,:,-1] = Exact_u[:,:,-1]

    # Create animation
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    fig.set_figheight(3.5)
    fig.set_figwidth(9)

    CS1 = axs[0].imshow(U_pred_plot[:,:,0], vmin = 0.0, vmax = 1.0, cmap=cm.nipy_spectral)
    fig.colorbar(CS1, ax=axs[0])

    CS2 = axs[1].imshow(Exact_plot[:,:,0].T, vmin = 0.0, vmax = 1.0, cmap=cm.nipy_spectral)
    fig.colorbar(CS2, ax=axs[1])

    def animate(i):

           axs[0].clear()
           axs[0].imshow(U_pred_plot[:,:,i], vmin = 0.0, vmax = 1.0, cmap=cm.nipy_spectral)
           axs[0].set_title("Predicted Cell Density - Time: " + str(round(t_plot[i][0],2)))

           axs[1].clear()
           axs[1].imshow(Exact_plot[:,:,i].T, vmin = 0.0, vmax = 1.0, cmap=cm.nipy_spectral)
           axs[1].set_title("Actual Cell Density - Time: " + str(round(t_plot[i][0],2)))

    interval = 1e+3*0.01 # 2nd number is the seconds
    ani = animation.FuncAnimation(fig, animate, num_plot_times, interval=interval, blit=False)
    if save_ani == 'on':
        ani.save('Animations/PI_Discrete_1D.gif')
    plt.show()

if printing == 'on':
    # Get D value and error
    final_D = np.exp(model_loss.layers[-1].get_weights()[0])[0]
    D_error = np.abs(D-final_D)/D*100
    # Get r value and error
    final_r = np.exp(model_loss.layers[-1].get_weights()[1])[0]
    r_error = np.abs(r-final_r)/r*100

    # Print results
    print('#######################################')
    print('Total Time Elapsed:', round(Total_Time/60, 2), 'minutes')
    print('Predicted D Value:', round(final_D, 5))
    print('Predicted r Value:', round(final_r, 5))
    print('Error in Predicted D:', str(round(D_error, 2)) + '%')
    print('Error in Predicted r:', str(round(r_error, 2)) + '%')
    print('#######################################')









































# end
