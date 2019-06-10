import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from unityagents import UnityEnvironment
from unity_env import UnityEnv

def make_env(file_name, wrapped = False):
    if wrapped:
        env = UnityEnv(environment_filename = file_name)
    else:
        env = UnityEnvironment(file_name = file_name)
    return env

def train(env, agent, n_episodes=300, max_t=1000):
    scores_deque = deque(maxlen=100)
    scores_list = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        agent.reset()
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            
            if np.any(dones):
                break
                
        score = np.mean(scores)
        scores_deque.append(score)
        scores_list.append(score)
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
        if np.mean(scores_deque) >= 31:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
        
    return scores_list

def train_wrapped(env, agent, n_episodes=400, max_t=1000, print_every=10):
    scores_deque = deque(maxlen=100)
    scores_list = []
    #max_score = -np.Inf
    epsilon = 1.0
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()
        scores = np.zeros(1) 
        for t in range(max_t):
            action = agent.act(state.reshape(1,33))
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            scores += reward
            if done:
                break 
        score = np.mean(scores)
        scores_deque.append(score)
        scores_list.append(score)
        #epsilon*=0.999
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
        if np.mean(scores_deque) >= 31:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
            
    return scores_list

def save_weights(agent, test_name):
    actor_name = 'weights_actor_'+str(test_name)+'.pth'
    critic_name = 'weights_critic_'+str(test_name)+'.pth'
    torch.save(agent.actor_local.state_dict(), actor_name)
    torch.save(agent.critic_local.state_dict(), critic_name)

def save_scores(scores, test_name):
    scores = pd.DataFrame({'Scores':scores})
    scores_name = str(test_name)+'-scores.csv'
    scores.to_csv(scores_name, sep=',')
    print('Learning curve saved in', scores_name)

def plot_smooth(scores, test_name, smoothing_window = 10):
    scores = pd.DataFrame({'Scores':scores})
    fig = plt.figure(figsize=(10,5))
    plt.grid(True)
    plt.style.use('seaborn-bright')
    rewards_smoothed = scores.rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plot_name = str(test_name)+'-plot.png'
    plt.savefig(plot_name)