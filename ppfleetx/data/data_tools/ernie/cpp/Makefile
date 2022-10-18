# CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
# CPPFLAGS += $(shell $(PYTHON_BIN) -m pybind11 --includes)

# LIBNAME = ernie_fast_index_map_helpers
# LIBEXT = .so

# default: $(LIBNAME)$(LIBEXT)

# %$(LIBEXT): %.cpp
# 	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@


CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
CPPFLAGS += $(shell $(PYTHON_BIN) -m pybind11 --includes)
LIBNAME = ernie_fast_index_map_helpers
LIBEXT = $(shell python3.7-config --extension-suffix)

default: $(LIBNAME)$(LIBEXT)

%$(LIBEXT): %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@