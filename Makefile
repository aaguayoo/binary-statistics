pre-commit:
	@git add .

compile:
	g++ -std=c++11 binaries_distributions.cpp distributions.cpp plot.cpp -I/usr/include/python3.9 -lpython3.9 -o main
