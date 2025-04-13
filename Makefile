FUNC := g++
copt := -c 
OBJ_DIR := ./bin/
FLAGS := -O3 -lm -g -Werror -fopenmp -mavx -mavx512f -mtune=native -march=native

CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix $(OBJ_DIR),$(notdir $(CPP_FILES:.cpp=.obj)))

all:
	$(FUNC) ./main.cpp -o ./main.exe $(FLAGS)

exact:
	$(FUNC) ./main.cpp -o ./main.exe $(FLAGS) -DEXACT

clean:
	rm -f ./*.exe