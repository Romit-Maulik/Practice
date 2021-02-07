#include <iostream>
#include <pybind11/embed.h> // everything needed for embedding

#include "Eigen/Dense"
#include <pybind11/eigen.h>
using Eigen::MatrixXd;

namespace py = pybind11;

int main(int argc, char *argv[])
{
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive
    py::print("Hello, World!"); // use the Python API

    //py::object tensorflow = py::module::import("tensorflow");
    auto os = py::module::import("os");
    auto sys = py::module::import("sys");

    // Loading module in working directory    
    py::module myModule = py::module::import("python_module");

    // Sending matrix using Eigen - becomes numpy array
    MatrixXd m(2,2);
    m(0,0) = 1;
    m(1,0) = 2;
    m(0,1) = 3;
    m(1,1) = 4;

    // Running function from module
    py::object result = myModule.attr("python_function")(m);
    MatrixXd res = result.cast<MatrixXd>();
    std::cout << "In C++ : \n" << res << std::endl;
    
    return 0;
}

