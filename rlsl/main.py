from collections import namedtuple
from env import Environment, SkipListWrapper
from itertools import count
from PIL import Image
import datetime
import helpers
import io
import math
import model
import numpy as np
import random
import skiplist
import sys
import time
import torch
import torch.distributions as distributions 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# From the PyTorch Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(object):
    def __init__(self, device, env, policy_net, target_net, params, metrics, fs):
        self.device = device
        
        self.BATCH_SIZE = params["batch_size"]
        self.GAMMA = params["gamma"]
        self.EPS_START = params["eps_start"]
        self.EPS_END = params["eps_end"]
        self.EPS_DECAY = params["eps_decay"]
        self.TARGET_UPDATE = params["target_update"]
        self.REPLAY_START = params["replay_start"]
 
        self.env = env

        self.policy_net = policy_net.to(self.device)

        self.target_net = target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(),
                                       lr=params["optim_lr"],
                                       alpha=params["optim_alpha"],
                                       eps=params["optim_eps"],
                                       centered=params["optim_centered"])
        
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=params["scheduler_milestones"],
                                                        gamma=params["scheduler_gamma"])
        
        self.memory = ReplayMemory(params["memory_size"])
        
        self.steps_done = 0
        self.episode_durations = []

        self.params = params
        self.metrics = metrics
        self.fs = fs

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].view(1, 1)
                q_value = float(q_values[0][q_values.argmax(1)])
                ts = datetime.datetime.fromtimestamp(time.time())
                metrics = []
                for i in range(6):
                    if action == i:
                        metrics.append({"MetricName": "action_%d" % (i), "Timestamp": ts, "Value": float(action)})
                    else:
                        metrics.append({"MetricName": "action_%d" % (i), "Timestamp": ts, "Value": 0 })
                metrics.append({"MetricName": "q_value", "Timestamp": ts, "Value": q_value })
                metrics.append({"MetricName": "eps_threshold", "Timestamp": ts, "Value": eps_threshold})
                self.metrics.write(metrics)
                return action
        else:
            print("EXPLORE!")
            action = torch.tensor([[random.randrange(self.params["h"] + 2)]], device=self.device, dtype=torch.long)
            q_values = self.policy_net(state)
            q_value = float(q_values[0][q_values.argmax(1)])
            ts = datetime.datetime.fromtimestamp(time.time())
            metrics = []
            for i in range(self.params["h"] + 2):
                if action == i:
                    metrics.append({"MetricName": "action_%d" % (i), "Timestamp": ts, "Value": float(action)})
                else:
                    metrics.append({"MetricName": "action_%d" % (i), "Timestamp": ts, "Value": 0 })
            metrics.append({"MetricName": "q_value", "Timestamp": ts, "Value": q_value })
            metrics.append({"MetricName": "eps_threshold", "Timestamp": ts, "Value": eps_threshold})
            self.metrics.write(metrics)
            return action

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        metrics = []
        ts = datetime.datetime.fromtimestamp(time.time())
        metrics.append({"MetricName": "lr", "Timestamp": ts, "Value": self.scheduler.get_lr()[0]})
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        metrics.append({"MetricName": "loss", "Timestamp": ts, "Value": loss.item() })
        self.metrics.write(metrics)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            metrics = []
            metrics.append({"MetricName": "grad_min", "Timestamp": ts, "Value": param.grad.data.min().item() })
            metrics.append({"MetricName": "grad_mean", "Timestamp": ts, "Value": param.grad.data.mean().item() })
            metrics.append({"MetricName": "grad_std", "Timestamp": ts, "Value": param.grad.data.std().item() })
            metrics.append({"MetricName": "grad_max", "Timestamp": ts, "Value": param.grad.data.max().item() })
            self.metrics.write(metrics)
        
        self.optimizer.step()      

    def train(self, num_episodes):
        for i_episode in range(num_episodes):
            print("EPISODE: %d" % (i_episode))
            # Initialize the environment and state
            self.env.reset()
            
            x = time.time()
            state = self.env.state()
            print("state time: %f" %(time.time() - x))

            total_reward = 0.0
            for t in count():
                print("EPISODE %d TURN %d" % (i_episode, t))
                # Select and perform an action
                action = self.select_action(state)
                x = time.time()
                _, reward, done, _ = self.env.step(action.item())
                print("step time %f" % (time.time() - x))
                print("REWARD: %f" % (reward))
                reward = torch.tensor([reward], device=self.device)

                x = time.time()
                ts = datetime.datetime.fromtimestamp(time.time())
                cpu_reward = reward.clone().cpu().item()
                total_reward += cpu_reward 
                self.metrics.write([{"MetricName":  "reward",
                                                    "Timestamp": ts,
                                                    "Value": cpu_reward
                                                  },{
                                                    "MetricName": "episode",
                                                    "Timestamp": ts,
                                                    "Value": i_episode
                                                  },{
                                                    "MetricName": "t",
                                                    "Timestamp": ts,
                                                    "Value": t
                                                  },{
                                                    "MetricName": "total_reward",
                                                    "Timestamp": ts,
                                                    "Value": total_reward
                                                  }])                
                print("CW Time: %d" % (time.time() - x))

                if t % self.params["img_interval"] == 0:
                    x = time.time()
                    data = state.clone().cpu().squeeze(0).squeeze(0).numpy()
                    img = Image.new('RGB', (1000,400), "black")
                    pixels = img.load()
                    for i in range(img.size[0]):    # For every pixel:
                        for j in range(4):
                            color = int(data[0][j][i] * 255)
                            for k in range(40):    
                                pixels[i,(j*40) + k] = (color, color, color) # Set the colour accordingly
                        if state[0][2][0][i] > 0:
                            for k in range(40):
                                pixels[i,(4*40) + k] = (100, 149, 237) # Set the colour accordingly
                        if i in env.peaks:
                            pixels[i,(4*40)] = (0, 255, 0)
                            pixels[i,(4*40) + 1] = (0, 255, 0)
                            pixels[i,(4*40) + 2] = (0, 255, 0)
                            pixels[i,(4*40) + 3] = (0, 255, 0)
                    access_min = data[1].min()
                    access_max = data[1].max()
                    for i in range(img.size[0]):
                        for j in range(4):
                            color = int(((data[1][j][i] - access_min) / (access_max - access_min)) * 255)
                            for k in range(40):    
                                pixels[i,(j*40) + k + 200] = (color, 0, 0) # Set the colour accordingly
                    ts_iso = datetime.datetime.fromtimestamp(time.time()).isoformat()
                    with io.BytesIO() as output:
                        img.save(output, "JPEG")
                        k = "pics/%s_%d_%d.jpeg" % (ts_iso, i_episode, t)
                        print("Saving to s3 %s" % (k))
                        #self.s3.put_object(Bucket="rlsl", Key=k, Body=output.getvalue())
                    print("Picture Time: %f" % (time.time() - x))

                if not done:
                    x = time.time()
                    next_state = self.env.state()
                    print("state time: %f" %(time.time() - x))
                else:
                    next_state = None
                
                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                x = time.time()
                if len(self.memory.memory) > self.REPLAY_START:
                    self.optimize_model()
                print("optimize_model time %f" % (time.time() - x))
                if done:
                    self.episode_durations.append(t + 1)
                    break

                # Update the target network
                if self.steps_done % self.TARGET_UPDATE == 0:
                    print("UPDATING TARGET NET!!!")
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            self.scheduler.step()
            ts_iso = datetime.datetime.fromtimestamp(time.time()).isoformat()
            with io.BytesIO() as output:
                torch.save(policy_net.state_dict(), output)
                #self.s3.put_object(Bucket="rlsl", Key="params/%s_policy_net_%d.state" % (ts_iso, i_episode), Body=output.getvalue())
            with io.BytesIO() as output:    
                torch.save(target_net.state_dict(), output)
                #self.s3.put_object(Bucket="rlsl", Key="params/%s_target_net_%d.state" % (ts_iso, i_episode), Body=output.getvalue())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = {"h": 4, 
              "key_count": 1000,
              "episodes": 1000,
              "batch_size": 32,
              "gamma": 0.99,
              "eps_start": 1,
              "eps_end": 0.1,
              "eps_decay": 20000,
              "target_update": 1000,
              "replay_start": 1000,
              "optim_lr": 0.005125,
              "optim_alpha": 0.95,
              "optim_eps": 0.01,
              "optim_centered": True,
              "scheduler_milestones": [50, 75],
              "scheduler_gamma": 0.1,
              "memory_size": 100000,
              "img_interval": 100,
              "s3_bucket": "rlsl",
              "cw_namespace": "rlsl"}

    keys = []
    kv_pairs = []
    for i in range(params["key_count"]):
        keys.append(str(i))
        kv_pairs.append((str(i), i))
    
    skip_list = SkipListWrapper(device, params["h"], kv_pairs)

    env = Environment(skip_list, keys)

    policy_net = model.Model(params["h"])
    target_net = model.Model(params["h"])

    metrics = helpers.CloudWatchMetrics(params["cw_namespace"])
    fs = helpers.S3FS(params["s3_bucket"])

    dqn = DQN(device, env, policy_net, target_net, params, metrics, fs)
    dqn.train(params["episodes"])
