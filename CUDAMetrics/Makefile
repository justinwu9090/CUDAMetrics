CUR=$(shell  pwd)
INC=-I/opt/cuda/include -I$(CUR)
#MORE=-D_GLIBCXX_DEBUG

all: main
	@echo done
main: 
	nvcc main.cu add.cu helper.cu $(INC) -o main 
test:
	clang++ $(MORE) -lclang -g -lboost_unit_test_framework $(INC) lib/Methods.cpp tests/Methods_test.cpp -o test
clean:
	rm -f add_cuda main
cleaninstall:
	sudo apt-get install libclang-dev libboost-all-dev
