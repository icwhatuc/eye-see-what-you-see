CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

%: %.cpp
	g++ $(CFLAGS) -o $@ $< $(LIBS)

eyedetect: eyedetect.cpp
	g++ $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -rf *.o webcam

