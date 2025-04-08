FUNC := g++
copt := -c 
OBJ_DIR := ./bin/
FLAGS := -O3 -lm -g -Werror -fopenmp -march=native -mtune=native # -fopt-info-vec-missed

CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix $(OBJ_DIR),$(notdir $(CPP_FILES:.cpp=.obj)))

all:
	$(FUNC) ./main.cpp -o ./main.exe $(FLAGS)

clean:
	rm -f ./*.exe