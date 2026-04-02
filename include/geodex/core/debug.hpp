#pragma once

#include <iostream>

#ifdef GEODEX_DEBUG
#define GEODEX_LOG(msg) std::cerr << "[geodex] " << msg << "\n"
#else
#define GEODEX_LOG(msg) ((void)0)
#endif
