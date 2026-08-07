"""Entraînement SAC et persistance complète d'un essai dans data/output/<run>."""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from src.assembly_env import TenonMortaiseEnv
from src.config import load_config, save_resolved_config

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/test1.yaml"); p.add_argument("--timesteps",type=int,default=500_000); p.add_argument("--seed",type=int,default=7); p.add_argument("--run",default=None); p.add_argument("--checkpoint-freq",type=int,default=50_000); p.add_argument("--device",default="auto"); args=p.parse_args()
    name=args.run or datetime.now().strftime("%Y%m%d_%H%M%S")
    out=Path("data/output")/name; out.mkdir(parents=True,exist_ok=False)
    # The run must remain reproducible even when source YAML files change.
    save_resolved_config(load_config(args.config), out/"config.yaml")
    env=Monitor(TenonMortaiseEnv(args.config), filename=str(out/"monitor.csv"), info_keywords=("success","position_error","rotation_error","force","torque","reward_position","reward_orientation","reward_progress","reward_force","reward_action","reward_success"))
    model=SAC("MlpPolicy",env,seed=args.seed,verbose=1,tensorboard_log=str(out/"tensorboard"),device=args.device,learning_starts=5_000,buffer_size=50_000,batch_size=256)
    callback=CheckpointCallback(args.checkpoint_freq,str(out/"checkpoints"),name_prefix="sac")
    model.learn(args.timesteps,callback=callback,progress_bar=True)
    model.save(out/"model"); env.close(); print(f"Essai sauvegardé: {out}")
if __name__=="__main__": main()
