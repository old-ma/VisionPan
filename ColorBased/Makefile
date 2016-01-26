# compiler selector
CC=g++
# Set executable file name
OUT = run
# Declaring the source code file
# Error if the files are not correctly located
C_SRC = main.c
CPP_SRC =  Algo.cpp
C_OBJ = $(patsubst %.c, %.o, $(C_SRC))
CPP_OBJ = $(patsubst %.cpp, %.o, $(CPP_SRC))
OBJ = $(C_OBJ) $(CPP_OBJ)

# Set gcc compiling flags

FLAGS_COMMON=-g -D_GNU_SOURCE  $(shell pkg-config opencv --cflags) 
CFLAGS = $(FLAGS_COMMON)
CPPFLAGS = $(FLAGS_COMMON) -w -std=c++0x 

LDFLAGS = $(shell pkg-config opencv --libs) 
# Final target - executable file
# Link all objects together
$(OUT): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

# Compile source code to object
# ATTENTION! This pattern-matching target cannot be the final target
%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)

%.o: %.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS)

# Clean all objects and final target
.PHONY: clean
clean:
	-@rm -f $(OUT) $(C_OBJ) $(CPP_OBJ)
