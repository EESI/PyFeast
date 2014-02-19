#include <string.h>
#include <errno.h>

#include "MIToolbox.h"

// a wrapper for calloc that checks if it's allocated
void *safe_calloc(size_t nelem, size_t elsize) {
	void *allocated = UNSAFE_CALLOC_FUNC(nelem, elsize);
	if(allocated == NULL) {
		fprintf(stderr, "Error: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}
	return allocated;
}
