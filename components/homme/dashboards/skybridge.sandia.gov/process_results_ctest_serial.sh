
cd /home/ikalash/nightlyHOMMEXXCDash
grep "Test   #" nightly_log_skybridgeHOMMEXXserial.txt >& results0
grep "Test  #" nightly_log_skybridgeHOMMEXXserial.txt >& results1
cat results0 results1 >& results11
grep "Test #" nightly_log_skybridgeHOMMEXXserial.txt >& results0
cat results11 results0 >& results1
echo "" >> results1
grep " tests failed" nightly_log_skybridgeHOMMEXXserial.txt >& results2
cat results1 results2 >& results3
grep "Total Test" nightly_log_skybridgeHOMMEXXserial.txt >& results4
cat results3 results4 >& results5
echo "" >> results5
grep "...   Passed" nightly_log_skybridgeHOMMEXXserial.txt >& results6
cat results5 results6 >& results_hommexx_serial 
rm results0 results1 results11 results2 results3 results4 results5 results6
