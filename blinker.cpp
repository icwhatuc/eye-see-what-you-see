#include <iostream>
#include <fstream>
#include <cv.h> 
#include <highgui.h>
#include <ctime>
#include <sys/time.h>

#define flip(x)		((x == '0')?x='1':x='0')
#define isOn(x)		(x == '1') // currently, when off -> bright pupil image
#define changeledstate()	{flip(arduino_state); arduino << arduino_state; arduino.flush();}

using namespace std;

float delay(float millisec)
{
	clock_t timesec;
	timesec = clock(); 
	while((clock() - timesec) < millisec*1000){}
	return millisec*1000;
}

void timing(bool start, string what="") {
	static struct timeval starttime, endtime;
	static long mtime, seconds, useconds;
	if (start) {
		gettimeofday(&starttime, NULL);
		cout << "timing " << what << endl;
	}
	else {
		gettimeofday(&endtime, NULL);
		seconds  = endtime.tv_sec  - starttime.tv_sec;
		useconds = endtime.tv_usec - starttime.tv_usec;

		mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;

		printf("Elapsed time: %ld milliseconds\n", mtime);
	}
}

main()
{
	char arduino_state = '0';
	ofstream arduino("/dev/ttyUSB0");
	float f;
	
	while(1)
	{
		flip(arduino_state);
		arduino << arduino_state;
		arduino.flush();
		
		delay(25);
	}
}
