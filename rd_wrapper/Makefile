CC = g++
CFLAGS = -Iglad/include -Iinclude -fpic -std=c++11
LDFLAGS = -lEGL -ldl -shared
TARGET = rd.so

$(TARGET): rd.o Renderer.o glad_egl.o glad.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

glad_egl.o: glad/src/glad_egl.c
	$(CC) -c -o $@ $< $(CFLAGS)

glad.o: glad/src/glad.c
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -f *.so *.o