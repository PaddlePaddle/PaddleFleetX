.PHONY : lib staticlib clean test

default : src.staticlib
install : src.install

BUILD_PREFIX ?= $(abspath ./build)
ABSBUILDDIR := $(abspath $(BUILD_PREFIX))

INSTALL_PREFIX ?= $(abspath ./install)
ABSINSTALLDIR := $(abspath $(INSTALL_PREFIX))

LIBTARGETS=src
TESTTARGETS=test

lib: ${LIBTARGETS:%=%.lib}
statliclib: ${LIBTARGETS:%=%.staticlib}
clean: ${LIBTARGETS:%=%.clean}
test: ${TESTTARGETS:%=%.test}

%.lib:
	${MAKE} -C $* lib BUILD_PREFIX=${ABSBUILDDIR}

%.staticlib:
	${MAKE} -C $* staticlib BUILD_PREFIX=${ABSBUILDDIR}

%.install:
	${MAKE} -C $* install BUILD_PREFIX=${ABSBUILDDIR} INSTALL_PREFIX=${ABSINSTALLDIR}

%.clean:
	${MAKE} -C $* clean BUILD_PREFIX=${ABSBUILDDIR}

%.test:
	${MAKE} -C $* test PREFIX=${ABSBUILDDIR}
