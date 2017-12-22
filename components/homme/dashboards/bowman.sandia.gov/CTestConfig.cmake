## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.
##
## # The following are required to submit to the CDash dashboard:
##   ENABLE_TESTING()
##   INCLUDE(CTest)

set(CTEST_PROJECT_NAME "HOMMEXX")
set(CTEST_NIGHTLY_START_TIME "00:00:00 EST")

set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "my.cdash.org")
set(CTEST_DROP_LOCATION "/submit.php?project=HOMMEXX")
set(CTEST_DROP_SITE_CDASH TRUE)


#set(CTEST_PROJECT_NAME "HOMMEXX")
#set(CTEST_NIGHTLY_START_TIME "00:00:00 EST")

#set(CTEST_DROP_METHOD "http")
#set(CTEST_DROP_SITE "cdash.sandia.gov")
#set(CTEST_DROP_LOCATION "/CDash-2-3-0/submit.php?project=HOMMEXX")
#set(CTEST_DROP_SITE_CDASH TRUE)
