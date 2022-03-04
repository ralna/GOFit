# Select compiler and options from below
############################
CC = gcc
CXX = g++
CFLAGS = -Wall -g -std=c11 -fopenmp
CXXFLAGS = -Wall -g -std=c++17 -fopenmp
############################
#CC = icc
#CXX = icpc
#CFLAGS = -Wall -g -std=c11 -qopenmp
#CXXFLAGS = -Wall -Wextra -g -std=c++17 -qopenmp
############################
#CC = icx
#CXX = icpx
#CFLAGS = -Wall -g -std=c11 -qopenmp
#CXXFLAGS = -Wall -g -std=c++17 -qopenmp
############################
INC = -I/usr/local/include/eigen3

# What to build
all: test_lhs test_numdiff test_regularisation test_multistart test_controller

# These rules build the packages
test_lhs: test_lhs.o lhs.o
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^
test_numdiff: test_numdiff.o numdiff.o
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^
test_regularisation: test_regularisation.o regularisation.o
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^
test_multistart: test_multistart.o lhs.o regularisation.o multistart.o
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^
test_controller: test_controller.o lhs.o regularisation.o multistart.o numdiff.o controller.o
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^

# The following rules build a source file from a variable
%.o: tests/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $<
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $<

# Rules to cleanup everything
clean:
	rm -f *.o
	rm -f test_lhs
	rm -f test_numdiff
	rm -f test_regularisation
	rm -f test_multistart
	rm -f test_controller
