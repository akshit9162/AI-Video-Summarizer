import argparse
from src.inference import run_inference

p=argparse.ArgumentParser()
p.add_argument("--video",required=True)
p.add_argument("--output",default="summary.mp4")
a=p.parse_args()

run_inference(a.video,a.output)
