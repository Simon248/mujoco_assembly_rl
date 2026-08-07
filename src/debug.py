"""Test 0: déplacement manuel et affichage des repères/wrench dans le viewer."""
from __future__ import annotations
import argparse
import numpy as np
from src.assembly_env import TenonMortaiseEnv
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/test0.yaml"); p.add_argument("--steps",type=int,default=300); args=p.parse_args()
 env=TenonMortaiseEnv(args.config,"human"); obs,_=env.reset(); print("obs=[erreur observée normalisée, wrench normalisé]",obs)
 for _ in range(args.steps):
  obs,_,terminated,truncated,info=env.step(np.array([0,0,-.15,0,0,0])); print(f"err={info['true_error']}, F={info['force']:.2f}N, T={info['torque']:.2f}Nm")
  if terminated or truncated: break
 env.close()
if __name__=="__main__": main()
