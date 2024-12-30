import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

#import new libraries
from keras.callbacks import TensorBoard
import os
from tqdm import tqdm


EPOCHS = 2
THRESHOLD = 45

#create list for metrics

rewards_control = []

#save metrics every n cycles
stats_cylcle = 1


callbacks = [tf.keras.callbacks.TensorBoard(log_dir='.\\DQN_exercise_CustomMetrics')]

class CustomTensorBoard(TensorBoard):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()
    
      
class DQN():
    def __init__(self, env_string,batch_size=64):
        self.memory = deque(maxlen=100000)
        self.env = gym.make(env_string)
        input_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        self.batch_size = batch_size
        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        alpha=0.01

        #for mac:
        #log_dir = "logs_summary/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        #create a variable to save TB logs
        self.tensorboard = CustomTensorBoard(log_dir='.\\DQN_Excerise_CustomMetrics')      


              
        # Create model
        self.model = Sequential()
        self.model.add(Dense(24, input_dim = input_size, activation = 'tanh'))
        self.model.add(Dense(48, activation = 'tanh'))
        self.model.add(Dense(action_size, activation = 'linear'))
        self.model.compile(loss = 'mse', optimizer = Adam(learning_rate = alpha), metrics = ['accuracy'])


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            #RL Module and Q-value prediction
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        #we now pass the variable "self.tensorboard" that inherits from CustomTensorBoard class
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size = len(x_batch), verbose = 1, callbacks = [self.tensorboard])
        
       
    def train(self):
        scores = deque(maxlen=100)
        avg_scores = []
        
        
        for e in tqdm(range(EPOCHS), ascii = False, unit = 'episodes'):
            #state = self.env.reset()

            agent.tensorboard.step = e
            episode_reward = 0


            state, info = self.env.reset()
            self.env.render()

            state = self.preprocess_state(state)
            done = False
            i = 0
            while not done:
                action = self.choose_action(state,self.epsilon)
                #next_state, reward, done, _ = self.env.step(action)
                next_state, reward, done, trunc, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.epsilon = max(self.epsilon_min, self.epsilon_decay*self.epsilon) # decrease epsilon
                i += 1

                #Rewards per episode
                episode_reward += reward
               
            
            scores.append(i)
            mean_score = np.mean(scores)
            avg_scores.append(mean_score)
            if mean_score >= THRESHOLD and e >= 100:
                print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                return avg_scores
            if e % 100 == 0:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

            self.replay(self.batch_size)


            #Custom metrics: AVG rewards
            rewards_control.append(episode_reward)
            if not e % stats_cylcle:
                average_rewards = sum(rewards_control)/len(rewards_control)
                max_reward = max(rewards_control)
                agent.tensorboard.update_stats(AVG_rewards = average_rewards, MAX_reward = max_reward, Epsilon_Decay = self.epsilon)       
        
        
        print('Did not solve after {} episodes 😞'.format(e))
        return avg_scores


env_string = 'CartPole-v0'
agent = DQN(env_string)

scores = deque(maxlen=100)

scores = agent.train()

plt.plot(scores)
plt.show()
agent.model.summary()
agent.env.close()

