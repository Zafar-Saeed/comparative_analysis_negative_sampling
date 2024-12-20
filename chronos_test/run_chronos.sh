#!/bin/bash
FILE=$1
USERNAME=zafarsaeed
PASSWORD=odk2Ii13sHAT63yZa6NNatooLnXYDLH5Z80XnNOSyzSZpvrU
HOSTNAME=hpc-gpu-1-4-1.recas.ba.infn.it 
PORT=54358
curl -u $USERNAME:$PASSWORD -L -H 'Content-Type: application/json' -X POST --data-binary "@$FILE" http://$HOSTNAME:$PORT/v1/scheduler/iso8601
