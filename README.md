# POLICY EVALUATION

## AIM
To develop a Python program to evaluate and improve a policy in the FrozenLake-v1 environment using policy iteration for maximizing cumulative reward in slippery terrain.

## PROBLEM STATEMENT
To implement policy iteration on the FrozenLake-v1 environment by performing policy evaluation and policy improvement repeatedly, and determine the optimal policy by comparing state-value functions, average return, and probability of reaching the goal.
## POLICY EVALUATION FUNCTION

```python
import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123);

pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk

def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)
def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)

env = gym.make('FrozenLake-v1')
P = env.env.P
init_state = env.reset()
goal_state = 15
LEFT, DOWN, RIGHT, UP = range(4)

P

state, reward, done, info = env.step(RIGHT)
print("state:{0} - reward:{1} - done:{2} - info:{3}".format(state, reward, done, info))

pi_frozenlake = lambda s: {
    0: RIGHT,
    1: DOWN,
    2: RIGHT,
    3: LEFT,
    4: DOWN,
    5: LEFT,
    6: RIGHT,
    7:LEFT,
    8: UP,
    9: DOWN,
    10:LEFT,
    11:DOWN,
    12:RIGHT,
    13:RIGHT,
    14:DOWN,
    15:LEFT #Stop
}[s]
print_policy(pi_frozenlake, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_frozenlake, goal_state=goal_state) * 100,
    mean_return(env, pi_frozenlake)))

pi_2 =lambda s: {
    0: RIGHT,
    1: RIGHT,
    2: DOWN,
    3: LEFT,
    4: DOWN,
    5: LEFT,
    6: DOWN,
    7: LEFT,
    8: RIGHT,
    9: DOWN,
    10: RIGHT,
    11: DOWN,
    12: RIGHT,
    13: RIGHT,
    14: RIGHT,
    15: LEFT
}[s]

print("Name: VARNIKA P     ")
print("Register Number:  212223240170       ")
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state) * 100,
    mean_return(env, pi_2)))

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P))

    while True:
        V = np.zeros(len(P))

        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))

        if np.max(np.abs(prev_V - V)) < theta:
            break

        prev_V = V.copy()

    return V

V1 = policy_evaluation(pi_frozenlake, P,gamma=0.99)
print_state_value_function(V1, P, n_cols=4, prec=5)

V2 = policy_evaluation(pi_2, P, gamma=0.99)
print_state_value_function(V2, P, n_cols=4, prec=5)

V1>=V2

if np.sum(V1) > np.sum(V2):
  print("Policy 1 is better.")
elif np.sum(V2) > np.sum(V1):
  print("Policy 2 is better.")
else:
  print("Policies are similar.")

```
## OUTPUT:

<img width="595" height="450" alt="image" src="https://github.com/user-attachments/assets/e833c064-f672-4cff-a492-8aa8de31ca48" />
<img width="666" height="110" alt="image" src="https://github.com/user-attachments/assets/c2b03ce1-6712-4878-aa3d-fa1103a59877" />
<img width="637" height="556" alt="image" src="https://github.com/user-attachments/assets/32e81c63-756d-4f26-a3bb-02afd1dfbc91" />
<img width="649" height="112" alt="image" src="https://github.com/user-attachments/assets/fc411885-5d25-41b2-ac89-ee857a52be8d" />
<img width="459" height="165" alt="image" src="https://github.com/user-attachments/assets/fff42184-964f-44f9-bc04-2a2d39fe1f43" />
<img width="442" height="157" alt="image" src="https://github.com/user-attachments/assets/14c27191-0b41-4a93-a2b4-a788ec85f2f6" />
<img width="557" height="92" alt="image" src="https://github.com/user-attachments/assets/02b7ddc6-eda2-4bd0-a814-513d5587ed7f" />
<img width="297" height="172" alt="image" src="https://github.com/user-attachments/assets/94e1935f-77bf-47f2-873f-64cc6f26c687" />


## RESULT:
Therefore, the optimal policy is obtained successfully using policy iteration in FrozenLake MDP.
