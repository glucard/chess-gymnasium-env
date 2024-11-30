from gymnasium.envs.registration import register

register(
    id="chess_gymnasium_env/ChessEnv-v0",
    entry_point="chess_gymnasium_env.envs:ChessEnv",
)
