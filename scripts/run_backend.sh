# Absolute path to this script. /home/user/bin/foo.sh
SCRIPT=$(readlink -f $0)
# Absolute path this script is in. /home/user/bin
SCRIPTPATH=`dirname $SCRIPT`

python "${SCRIPTPATH}/../src/ga_prompt_llm/backend/hypercycle_server.py"