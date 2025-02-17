#!/bin/bash -
python3 -u main.py NT2 DBLP --runs 5 --epochs 10000 --heads 4 --hidden 34 --n-layers 5 --dropout 0.07 --input-dropout 0.12 --with-node-proxy --patience 1000
python3 -u main.py NT2 AMinerAuthor --runs 5 --epochs 10000 --heads 5 --hidden 13 --n-layers 1 --dropout 0.06 --input-dropout 0.07 --with-node-proxy --patience 1000
python3 -u main.py NT2 emailEnron --runs 5 --epochs 10000 --heads 7 --hidden 11 --n-layers 3 --dropout 0.03 --input-dropout 0.00 --with-edge-proxy --patience 1000
python3 -u main.py NT2 emailEu --runs 5 --epochs 10000 --heads 6 --hidden 13 --n-layers 2 --dropout 0.02 --input-dropout 0.03 --patience 1000
python3 -u main.py NT2 StackOverflowBiology --runs 5 --epochs 10000 --heads 6 --hidden 16 --n-layers 5 --dropout 0.04 --input-dropout 0.11 --with-node-proxy --patience 1000
python3 -u main.py NT2 StackOverflowPhysics --runs 5 --epochs 10000 --heads 5 --hidden 14 --n-layers 9 --dropout 0.00 --input-dropout 0.03 --with-edge-proxy --patience 1000
