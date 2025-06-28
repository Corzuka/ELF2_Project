#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

int init_uart(int fd)
{
	struct termios newtio, oldtio;
	if (tcgetattr(fd, &oldtio) != 0)
	{
		perror("tcgetattr");
		return -1;
	}
	bzero(&newtio, sizeof(newtio));
	newtio.c_cflag |= CLOCAL | CREAD;
	newtio.c_cflag &= ~CSIZE;
	newtio.c_cflag |= CS8;
	newtio.c_cflag &= ~PARENB;
	cfsetispeed(&newtio, B9600);
	cfsetospeed(&newtio, B9600);
	newtio.c_cflag &= ~CSTOPB;
	newtio.c_cc[VTIME] = 0;
	newtio.c_cc[VMIN] = 0;
	tcflush(fd, TCIFLUSH);
	if ((tcsetattr(fd, TCSANOW, &newtio)) != 0)
	{
		perror("com set error");
		return -1;
	}
	printf("set done!\n");
	return 0;
}

int uart_send_frame(int fd,const unsigned char *p_send_buff, const int count)
{
	int Result = 0;

	Result = write(fd, p_send_buff, count);
	if (Result == -1)
	{
		perror("write");
		return 0;
	}
	return Result;
}

int uart_read_frame(int fd,unsigned char *p_receive_buff, const int count , int timeout_data)
{
	int nread = 0;
	fd_set rd;
	int retval = 0;
	struct timeval timeout = {0, timeout_data*1000};

	FD_ZERO(&rd);
	FD_SET(fd, &rd);
	memset(p_receive_buff, 0x0, count);
	//printf("等待串口数据...\n");
	retval = select(fd + 1, &rd, NULL, NULL, &timeout);
	switch (retval)
	{
		case 0:
			nread = 0;
			//printf("超时，无数据\n");
			break;
		case -1:
			printf("select%s\n", strerror(errno));
			nread = -1;
			break;
		default:
			nread = read(fd, p_receive_buff, count);
			//printf("读到 %d 字节数据: ", nread);
            		for (int i = 0; i < nread; i++) {
                		printf("%02X ", p_receive_buff[i]);
            		}
            		printf("\n");
			break;
	}
	return nread;
}

int main(int argc , char **argv)
{
	int fd;
	char buf[10];
	pid_t pid;

	fd = open(argv[1], O_RDWR | O_NOCTTY | O_NDELAY );
	init_uart(fd);

	while(1)
	{
		//printf("开始读取帧...\n");
		uart_read_frame(fd,buf,10,1000);
		//printf("读取完成\n");
		if(buf[0] != '\0')
		{
			// 检测到特定指令则退出
            		if (strstr(buf, "begin")) {
                		printf("检测到开始巡检指令，退出程序\n");
                		close(fd);
                		exit(0);  // 正常退出
            		}
		}
		usleep(100000);
	}

	return 0;
}
