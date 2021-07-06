/*
Compile with
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
*/


#include <pybind11/pybind11.h>
namespace py = pybind11;


double add(double i, double j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");

    m.attr("the_answer") = 42;
    py::object world = py::cast("World");
    m.attr("what") = world;
}