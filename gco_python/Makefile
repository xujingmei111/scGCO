CXX=g++
CFLAGS=-fPIC
INC=-Igco_source -Igco

all: gco_source/gco.so

gco_source/gco.so: \
    gco_source/LinkedBlockList.o gco_source/graph.o gco_source/maxflow.o \
        gco_source/GCoptimization.o
	$(CXX) -shared $(CFLAGS) gco_source/LinkedBlockList.o \
	    gco_source/graph.o \
	    gco_source/maxflow.o \
	    gco_source/GCoptimization.o -o gco_source/gco.so

gco_source/LinkedBlockList.o: \
    gco_source/LinkedBlockList.cpp \
        gco_source/LinkedBlockList.h
	$(CXX) $(CFLAGS) $(INC) \
	    -c gco_source/LinkedBlockList.cpp \
	    -o gco_source/LinkedBlockList.o

gco_source/graph.o: \
    gco_source/graph.cpp gco_source/graph.h gco_source/block.h
	$(CXX) $(CFLAGS) $(INC) \
	    -c -x c++ gco_source/graph.cpp \
	    -o gco_source/graph.o

gco_source/maxflow.o: \
    gco_source/block.h gco_source/graph.h gco_source/maxflow.cpp
	$(CXX) $(CFLAGS) $(INC) \
	    -c -x c++ gco_source/maxflow.cpp \
	    -o gco_source/maxflow.o

gco_source/GCoptimization.o: \
    gco_source/GCoptimization.cpp gco_source/GCoptimization.h \
        gco_source/LinkedBlockList.h gco_source/energy.h gco_source/graph.h \
        gco_source/graph.o gco_source/maxflow.o
	$(CXX) $(CFLAGS) $(INC) \
	    -c gco_source/GCoptimization.cpp \
	    -o gco_source/GCoptimization.o

clean:
	rm -f *.o gco_source/*.o 

rm:
	rm -f *.o *.so gco_source/*.o *.zip 
