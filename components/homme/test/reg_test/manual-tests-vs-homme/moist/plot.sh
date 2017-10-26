#!/bin/bash -f

#source ~/.bash_profile
export NCARG_ROOT=/projects/ccsm/ncl


#plot homme r0
#/projects/ccsm/ncl/bin/ncl 'fnames="movies/homme-r0-jw_baroclinic"' zeta.ncl
#plot homme r3
/projects/ccsm/ncl/bin/ncl 'fnames="movies/homme72-r3-jw_baroclinic"' zeta.ncl
#plot xx-f r0
#/projects/ccsm/ncl/bin/ncl 'fnames="movies/f-r0-jw_baroclinic"' zeta.ncl
#plot xx-f r3
/projects/ccsm/ncl/bin/ncl 'fnames="movies/f72-r3-jw_baroclinic"' zeta.ncl
#plot xx-c r0
#/projects/ccsm/ncl/bin/ncl 'fnames="movies/cxx-r0-jw_baroclinic"' zeta.ncl
#plot xx-c r3
#/projects/ccsm/ncl/bin/ncl 'fnames="movies/cxx-r3-jw_baroclinic"' zeta.ncl






