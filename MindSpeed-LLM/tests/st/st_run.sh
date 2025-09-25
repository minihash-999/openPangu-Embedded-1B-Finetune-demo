# step 1: define dir
BASE_DIR=$(dirname "$(readlink -f "$0")")
export PYTHONPATH=$BASE_DIR:$PYTHONPATH

SHELL_SCRIPTS_DIR="$BASE_DIR/shell_scripts"
BASELINE_DIR="$BASE_DIR/baseline_results"
EXEC_PY_DIR=$(dirname "$BASE_DIR")

GENERATE_LOG_DIR=/data/run_logs
GENERATE_JSON_DIR=/data/run_jsons

rm -rf $GENERATE_LOG_DIR/*
rm -rf $GENERATE_JSON_DIR/*


# step 2: running scripts and execute `test_ci_pipeline.py`
for test_case in "$SHELL_SCRIPTS_DIR"/*.sh; do
    file_name=$(basename "${test_case}")
    echo "Running $file_name..."
    file_name_prefix=$(basename "${file_name%.*}")
    echo "$file_name_prefix"

    # create empty json file to receive the result parsered from log
    touch "$GENERATE_JSON_DIR/$file_name_prefix.json"
    
    # if executing the shell has failed, then just exit, no need to compare.
    bash $test_case | tee "$GENERATE_LOG_DIR/$file_name_prefix.log"
    SCRIPT_EXITCODE=${PIPESTATUS[0]}
    if [ $SCRIPT_EXITCODE -ne 0 ]; then
        echo "Script has failed. Exit!"
        exit 1
    fi
    # begin to execute the logic of compare
    pytest -x $EXEC_PY_DIR/test_tools/test_ci_st.py \
        --baseline-json $BASELINE_DIR/$file_name_prefix.json \
        --generate-log $GENERATE_LOG_DIR/$file_name_prefix.log \
        --generate-json $GENERATE_JSON_DIR/$file_name_prefix.json
    PYTEST_EXITCODE=$?
    if [ $PYTEST_EXITCODE -ne 0 ]; then
        echo "$file_name_prefix compare to baseline has failed, check it!"
        exit 1
    else
        echo "Pretrain $file_name_prefix execution success."
    fi

done

