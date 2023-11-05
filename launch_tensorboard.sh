#!/usr/bin/env bash

#dir = `ls runs | sort | tail -1`
echo "Checkpoint dir: " `ls runs | sort | tail -1`

tensorboard --logdir=runs/`ls runs | sort | tail -1`