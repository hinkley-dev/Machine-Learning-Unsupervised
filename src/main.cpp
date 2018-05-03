// This file is licensed under the CC0 license (See http://creativecommons.org/publicdomain/zero/1.0/).
// And just to make sure you get the idea, it is also licensed under the WTFPL (See http://en.wikipedia.org/wiki/WTFPL).

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <exception>
#include <string>
#include <memory>
#include "error.h"
#include "string.h"
#include "image.h"

using std::cout;
using std::cerr;
using std::string;
using std::auto_ptr;

void test()
{
	MyImage i;
	i.resize(255, 255);
	for(int y = 0; y < 255; y++)
	{
		for(size_t x = 0; x < 255; x++)
		{
			unsigned int color = 0xff000000 | (y << 16) | (x << 8) | y;
			i.setPixel(x, y, color);
		}
	}
	i.savePng("test.png");
}

int main(int argc, char *argv[])
{
	enableFloatingPointExceptions();
	int ret = 1;
	try
	{
		test();
		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	cout.flush();
	cerr.flush();
	return ret;
}
