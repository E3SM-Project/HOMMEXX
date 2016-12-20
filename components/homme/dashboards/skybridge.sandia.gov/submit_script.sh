#!/bin/sh

UPLOAD_FILES="True"

BASE_DIR=/home/ikalash/hommexxCDash/SkybridgeHOMMEXXSerial
SCRIPT_DIR=/home/ikalash/hommexxCDash/SkybridgeHOMMEXXSerial
DOWNLOAD_DIR=$BASE_DIR/Results 

CDASH_SITE="http://my.cdash.org/submit.php?project=HOMMEXX"

#export PATH=/usr/bin:/bin:/sbin:/usr/sbin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=/home/ikalash/hommexxCDash/SkybridgeHOMMEXXSerial/submit.log

cd $BASE_DIR
rm $LOG_FILE 

echo "Running CDash post at $now" > $LOG_FILE
echo >> $LOG_FILE


if [ "$UPLOAD_FILES" ]; then

# curl the Project.xml file to the CDash site
  echo "Sending Project.xml to CDash site: $CDASH_SITE" >> $LOG_FILE
  echo >> $LOG_FILE
  curl -T $SCRIPT_DIR/Project.xml $CDASH_SITE >> $LOG_FILE 2>&1
  echo >> $LOG_FILE


# curl the files to the CDash site

  if [ -d "$DOWNLOAD_DIR" ]; then

   for files in $DOWNLOAD_DIR/*; do
      echo "Sending $files to CDash site: $CDASH_SITE" >> $LOG_FILE
      curl -T $files $CDASH_SITE >> $LOG_FILE 2>&1
      echo >> $LOG_FILE
   done
  fi

fi

echo >> $LOG_FILE
echo "Done!!!" >> $LOG_FILE
