CC ?= gcc
CFLAGS ?= -O3 -shared -fPIC

all: prism/libfastcodecs.so prism/libexactacc.so

prism/libfastcodecs.so: prism/fastcodecs.c
	$(CC) $(CFLAGS) -o $@ $<

prism/libexactacc.so: prism/exactacc.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f prism/*.so

.PHONY: all clean
