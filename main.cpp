// main.cpp

// don't forget to include out neural network
#include "NeuralNetwork.h"

//... data generator code here
typedef std::vector<RowVector*> data;

void ReadCSV(std::string filename, std::vector<RowVector*>& data)
{
	data.clear();
	std::ifstream file(filename);
	std::string line, word;

	// determine number of columns in file
	std::getline(file, line);
	std::stringstream ss(line);
	std::vector<Scalar> parsed_vec;
	while (std::getline(ss, word, ',')) {
		parsed_vec.push_back(Scalar(std::stof(&word[0])));
	}
	int cols = parsed_vec.size();
	data.push_back(new RowVector(cols));
	for (int i = 0; i < cols; i++) {
		data.back()->coeffRef(1, i) = parsed_vec[i];
	}

	// read the file
	if (file.is_open()) {
		while (std::getline(file, line)) {
			std::stringstream ss(line);
			data.push_back(new RowVector(1, cols));
			int i = 0;
			while (std::getline(ss, word, ',')) {
				data.back()->coeffRef(i) = Scalar(std::stof(&word[0]));
				i++;
			}
			std::cout << "filename= " << filename << ", push_back value= " << word << std::endl;
		}
	}
}

void genData(std::string filename)
{
	std::cout << "Opening filename:" << filename << " -in and -out" << std::endl;
	std::string infilename = filename + "-in.txt";
	std::string outfilename = filename + "-out.txt";

	std::ifstream infile(infilename);
	if (!infile.good()) {
		std::ofstream file1(infilename, std::ofstream::out);
		std::ofstream file2(outfilename, std::ofstream::out);
		for (int r = 0; r < 1000; r++) {
			Scalar x = rand() / Scalar(RAND_MAX);
			Scalar y = rand() / Scalar(RAND_MAX);
			std::cout << "x=" << x << ", y=" << y << std::endl;
			file1 << x << "," << y << std::endl;
			file2 << 2 * x + 10 + y << std::endl;
		}
		file1.close();
		file2.close();
	}
	
}


int main()
{
	std::cout << "NeuralNetwork test ongoing \n" << std::endl;
    genData("test");

	data in_dat, out_dat;
	std::vector<int> nntopology = { 2, 3, 1 };

    ReadCSV("test-in.txt", in_dat);
    ReadCSV("test-out.txt", out_dat);

	NeuralNetwork n(nntopology);
    n.train(in_dat, out_dat);

    return 0;
}
