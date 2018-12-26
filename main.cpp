// C++ headers
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <memory>
#include <stdio.h>
#include <sstream>
#include <string>
#include <time.h>
#include <vector>
#include<unordered_map>

// My headers
#include "Headers.h"

int main() { // Start of program
    /**
    Docs auto generated with DOxygen
    Program starts here
    */

	// Add runs here - Cavity flow class
	std::shared_ptr< Cavity_flow > run1( new Cavity_flow( 10.0, 24, 24, 2.0 ) );
	run1->init_domain();
	run1->artificial_compressibility();
	printf("So far so good\n");

}
