## STLport is the location of STLport ##
## the library can be download from 
## http://www.stlport.org/
## location of STLport library without '/' at the end
STLport=./STLport

## SVMsrc is the SVMlight source code 
## which can be downloaded from
## http://svmlight.joachims.org/
## location of SVMlight without '/' at the end
SVMlight=./SVMlight

## tclap library
## included in the package, shouldn't be edited
TCLAP=./tclap-1.2.0

## g++/gcc options for code optimization
GNUREC=-O3 -ffast-math -funroll-all-loops -fpeel-loops -ftracer -funswitch-loops -funit-at-a-time -pthread -Wno-deprecated -std=c++0x
GO=$(GNUREC)

## these lines shouldn't be edited,
SVMsrc=svm_hideo.c svm_common.c svm_learn.c 
SVMh=svm_common.h svm_learn.h kernel.h

## g++ 
CC=g++ $(GO)
GCC=gcc $(GO)

INCLUDES=-I$(TCLAP)/include/ -I$(STLport)/include/stlport
LIBS=-L$(STLport)/lib
CFLAGS=-c $(INCLUDES) 
LDFLAGS=$(LIBS) 
INLIBS=-lstlport -lgcc_s -lpthread -lc -lm
L2Psrc=allPhonemeSet.cpp weightWF.cpp miraPhrase.cpp phraseModel.cpp
SOURCES=$(SVMsrc) $(L2Psrc)
SVMobj=$(SVMsrc:.c=.o)
L2Pobj=$(L2Psrc:.cpp=.o)
OBJECTS=$(SVMobj) $(L2Pobj)
EXECUTABLE=directlpCopy

all: SVM_light $($SOURCES) $(EXECUTABLE)

SVM_light: $(SVMsrc) $(SVMh)

$(SVMsrc): 
	## copy over needed SVM-light source codes ##
	cp $(SVMlight)/$@ $@

$(SVMh): 
	## modified SVM-light head files ##
	sed "1 i #ifdef __cplusplus \nextern \"C\" { \n#endif"  $(SVMlight)/$@ > $@-tmp
	sed "$$ a #ifdef __cplusplus \n} \n#endif" $@-tmp > $@
	rm $@-tmp
	

$(EXECUTABLE):	$(OBJECTS) 
	## linking ##
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@ $(INLIBS)

.cpp.o:
	## compiling C++ codes ##
	$(CC) $(CFLAGS) $< -o $@ 
.c.o:
	## compiling C codes ##
	$(GCC) $(CFLAGS) $< -o $@

clean:	
	rm -f $(EXECUTABLE) $(OBJECTS) $(SVMsrc) $(SVMh)
