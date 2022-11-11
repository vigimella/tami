SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}

function usage {
    echo "USAGE: $0 [--quantum]"
    exit
}

QUANTUM=0

for arg in "$@"; do
    if [ "$arg" == "--quantum" ]; then
        QUANTUM=1
    else
        echo "ERROR! Incorrect parameter '$arg'"
        usage 
    fi 
done

if (($QUANTUM)); then
   URLTAMI="https://martellone.iit.cnr.it/index.php/s/Gam4Hw4Hia49yBX/download/tami_exp_quantumV1.0.tar"
else
   URLTAMI="https://martellone.iit.cnr.it/index.php/s/8qA8jtiK5SYtMk9/download/tami_expV1.0.tar"
fi


#docker save -o <path for generated tar file> <image name>
echo "Downloading TAMI image from ocsdev cloud"
wget ${URLTAMI}
echo "Loading TAMI image in local docker instance (it may take a while...)"
NAMETAMI=$(basename ${URLTAMI})
docker load -i ${NAMETAMI}

if [ $? -eq 0 ]; then
   echo "Load completed, you can now run the script 'run_container.sh'"
   rm ${NAMETAMI}
else
   echo "Load failed..."
fi