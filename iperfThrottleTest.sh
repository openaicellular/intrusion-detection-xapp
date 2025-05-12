#!/bin/bash

# Namespace, server IP, and port
NS="ue1"
SERVER_IP="172.16.0.1"
PORT="5006"

# Bandwidth toggle values
BW1="20M"
BW2="1M"

TOGGLE_INTERVAL=15

CURRENT_BW=$BW1

echo "Starting iperf3 bandwidth toggle test (toggle every ${TOGGLE_INTERVAL}s)..."

# Run indefinitely
while true; do
    echo "Running iperf3 client with bandwidth: $CURRENT_BW"
    sudo ip netns exec $NS iperf3 -c $SERVER_IP -p $PORT -i 1 -t $TOGGLE_INTERVAL -R -b $CURRENT_BW

    # Toggle bandwidth
    if [ "$CURRENT_BW" == "$BW1" ]; then
        CURRENT_BW=$BW2
    else
        CURRENT_BW=$BW1
    fi

    sleep 0.5
done
