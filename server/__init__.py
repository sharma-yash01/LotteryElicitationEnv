"""LotteryElicitationEnv server components."""

try:
    from env.lottery_env import LotteryElicitationEnvironment
except ImportError:
    from ..env.lottery_env import LotteryElicitationEnvironment

__all__ = ["LotteryElicitationEnvironment"]
