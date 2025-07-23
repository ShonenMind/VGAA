
import procgen #importred procgen
import gym  #imported gym


def main():
   env = gym.make("procgen-coinrun-v0")
   obs = env.reset()
   print(f"initial observation shape: {obs.shape}")


   done = False
   steps = 0
   while not done and steps < 100:
       action = env.action_space.sample()  # random action
       obs, reward, done, info = env.step(action)
       steps += 1
       print(f"Step: {steps}, Reward: {reward}, Done: {done}")


   env.close()


if __name__ == "__main__":
   main()
