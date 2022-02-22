# Select compiler and options from below
############################
CC = gcc
CXX = g++
CFLAGS = -Wall -g -std=c11
CXXFLAGS = -Wall -g -std=c++17
############################
#CC = icc
#CXX = icpc
#CFLAGS = -Wall -g -std=c11
#CXXFLAGS = -Wall -Wextra -g -std=c++17
############################
#CC = icx
#CXX = icpx
#CFLAGS = -Wall -g -std=c11
#CXXFLAGS = -Wall -g -std=c++17
############################
INC = -I/usr/local/include/eigen3

# What to build
all: test_regularisation

# These rules build the packages
test_regularisation: test_regularisation.o regularisation.o
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^

# The following rules build a source file from a variable
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $<

# Rules to cleanup everything
clean:
	rm -f *.o
	rm -f test_regularisation
