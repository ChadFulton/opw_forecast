gfortran -c truncated_normal.f90 -o truncated_normal.o
gfortran simulate.f95 truncated_normal.o \
-llapack -lblas \
-fcheck=all -Wextra -Wall \
-o simulate