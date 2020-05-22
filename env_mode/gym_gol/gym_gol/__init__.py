from gym.envs.registration import register

register(id='gol-v0', entry_point='gym_gol.envs:GolEnv')
register(id='gol-extrahard-v0', entry_point='gyn_gol.envs:GolExtraHardEnv')

