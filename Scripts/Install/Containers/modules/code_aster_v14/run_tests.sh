NON_ROOT_USER=ibsim
# NON_ROOT_USER=aster

if [ -z "$ASRUN" ]; then
    ASRUN=/home/$NON_ROOT_USER/aster/bin/as_run
fi

TEST_DIR=/home/$NON_ROOT_USER/shared/test

mkdir -p $TEST_DIR
cd $TEST_DIR
$ASRUN --list --all --output=$TEST_DIR/testcases
# $ASRUN --list --all --filter='"parallel" in testlist' --output=$TEST_DIR/testcases

for testcase in `cat $TEST_DIR/testcases`
do
    echo "Working on $testcase..."
    $ASRUN --test $testcase $TEST_DIR >> $TEST_DIR/screen
done

$ASRUN --diag --only_nook --astest_dir=$TEST_DIR > $TEST_DIR/diag
