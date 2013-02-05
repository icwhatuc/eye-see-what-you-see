CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

%: %.cpp
	g++ $(CFLAGS) -g -o $@ $< $(LIBS)

eyedetect: eyedetect.cpp
	g++ $(CFLAGS) -g -o $@ $< $(LIBS)

clean:
	rm -rf *.o webcam

