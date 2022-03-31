# Select compiler and options from below
############################
CXX = g++
CXXFLAGS = -Wall -g -std=c++17 -fopenmp
############################
#CXX = icpc
#CXXFLAGS = -Wall -Wextra -g -std=c++17 -qopenmp
############################
#CXX = icpx
#CXXFLAGS = -Wall -g -std=c++17 -qopenmp
############################
INC = -I/usr/local/include/eigen3
#INC = -I/usr/include/eigen3
#INC = -I/home/fowkes/Code/eigen3

# What to build
all: test_lhs test_findiff test_regularisation test_multistart test_alternating

# These rules build the packages
test_lhs: test_lhs.o lhs.o
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^
test_findiff: test_findiff.o findiff.o
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^
test_regularisation: test_regularisation.o regularisation.o findiff.o
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^
test_multistart: test_multistart.o lhs.o regularisation.o multistart.o findiff.o
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^
test_alternating: test_alternating.o lhs.o regularisation.o multistart.o findiff.o alternating.o
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
	rm -f test_findiff
	rm -f test_regularisation
	rm -f test_multistart
	rm -f test_alternating
