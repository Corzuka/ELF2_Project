CPP=aarch64-linux-gcc
sound_demo:soundapp.o
	$(CPP)  -Wall   soundapp.o -o     soundapp
soundapp.o:soundapp.c
	$(CPP)  -c      -Wall   soundapp.c  -o      soundapp.o
clean:
	$(RM) *.o       soundapp
