from chess_gymnasium_env.envs.chess import ChessEnv
from sb3_contrib.common.wrappers import ActionMasker

def action_mask_fn(env):
    return env.unwrapped._get_action_mask()

def make_env():
    def _init():
        env = ChessEnv(render_mode="human")
        env = ActionMasker(env, action_mask_fn)
        return env
    return _init