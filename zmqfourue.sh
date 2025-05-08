#!/bin/sh

SLEEPINT=5;

export SS_XAPP=`kubectl get svc -n ricxapp --field-selector metadata.name=service-ricxapp-ss-rmr -o jsonpath='{.items[0].spec.clusterIP}'`
if [ -z "$SS_XAPP" ]; then
    export SS_XAPP=`kubectl get svc -n ricxapp --field-selector metadata.name=service-ricxapp-ss-rmr -o jsonpath='{.items[0].spec.clusterIP}'`
fi
if [ -z "$SS_XAPP" ]; then
    echo "ERROR: failed to find drl-ss nbi service; aborting!"
    exit 1
fi

echo SS_XAPP=$SS_XAPP ; echo

echo Listing NodeBs: ; echo
curl -i -X GET http://${SS_XAPP}:8000/v1/nodebs ; echo ; echo
echo Listing Slices: ; echo
curl -i -X GET http://${SS_XAPP}:8000/v1/slices ; echo ; echo
echo Listing Ues: ; echo
curl -i -X GET http://${SS_XAPP}:8000/v1/ues ; echo ; echo

sleep $SLEEPINT

echo "Creating NodeB (id=1):" ; echo
curl -i -X POST -H "Content-type: application/json" -d '{"type":"eNB","id":411,"mcc":"001","mnc":"01"}' http://${SS_XAPP}:8000/v1/nodebs ; echo ; echo
echo Listing NodeBs: ; echo
curl -i -X GET http://${SS_XAPP}:8000/v1/nodebs ; echo ; echo

sleep $SLEEPINT

echo "Creating Slice (name=fast)": ; echo
curl -i -X POST -H "Content-type: application/json" -d '{"name":"fast","allocation_policy":{"type":"proportional","share":1024}}' http://${SS_XAPP}:8000/v1/slices ; echo ; echo
echo Listing Slices: ; echo
curl -i -X GET http://${SS_XAPP}:8000/v1/slices ; echo ; echo

sleep $SLEEPINT

echo "Creating Slice (name=secure_slice)": ; echo
curl -i -X POST -H "Content-type: application/json" -d '{"name":"secure_slice","allocation_policy":{"type":"proportional","share":0}}' http://${SS_XAPP}:8000/v1/slices ; echo ; echo
echo Listing Slices: ; echo
curl -i -X GET http://${SS_XAPP}:8000/v1/slices ; echo ; echo

sleep $SLEEPINT

echo "Binding Slice to NodeB (name=fast):" ; echo
curl -i -X POST http://${SS_XAPP}:8000/v1/nodebs/enB_macro_001_001_0019b0/slices/fast ; echo ; echo
echo "Getting NodeB (name=enB_macro_001_001_0019b0):" ; echo
curl -i -X GET http://${SS_XAPP}:8000/v1/nodebs/enB_macro_001_001_0019b0 ; echo ; echo

sleep $SLEEPINT

echo "Binding Slice to NodeB (name=secure_slice):" ; echo
curl -i -X POST http://${SS_XAPP}:8000/v1/nodebs/enB_macro_001_001_0019b0/slices/secure_slice ; echo ; echo
echo "Getting NodeB (name=enB_macro_001_001_0019b0):" ; echo
curl -i -X GET http://${SS_XAPP}:8000/v1/nodebs/enB_macro_001_001_0019b0 ; echo ; echo

sleep $SLEEPINT

# ✅ Dynamically create UEs and bind to slice
UE_DB="/root/.config/srslte/user_db.csv"

while IFS=',' read -r name auth imsi key op_type opc amf sqn qci ip_alloc; do
    # Skip comments and blank lines
    echo "$name" | grep -q '^#' && continue
    [ -z "$imsi" ] && continue

    echo "Creating Ue (ue=$imsi)" ; echo
    curl -i -X POST -H "Content-type: application/json" -d "{\"imsi\":\"$imsi\"}" http://${SS_XAPP}:8000/v1/ues ; echo ; echo

    sleep $SLEEPINT

    echo "Binding Ue (imsi=$imsi):" ; echo
    curl -i -X POST http://${SS_XAPP}:8000/v1/slices/fast/ues/$imsi ; echo ; echo

    sleep $SLEEPINT
done < "$UE_DB"

# Final UE and slice listing
echo Listing Ues: ; echo
curl -i -X GET http://${SS_XAPP}:8000/v1/ues ; echo ; echo

echo "Getting Slice (name=fast):" ; echo
curl -i -X GET http://${SS_XAPP}:8000/v1/slices/fast ; echo ; echo

