#include <caffe/caffe.hpp>
#include <memory>
#include "caffe/layers/memory_data_layer.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <climits>
#include <arpa/inet.h>

std::ostream &output = std::cout;

void load_mnist(const char *filenameImages, const char *filenameLabels,
				std::vector<cv::Mat> &images, std::vector<int> &labels)
{
	int magicImages;
	int magicLabels;
	int countImages;
	int countLabels;
	int width;
	int height;

	std::ifstream fileImages(filenameImages, std::ios::in | std::ios::binary);
	if (!fileImages)
	{
		output << "Images bin file not found. : " << filenameImages << std::endl;
	}

	std::ifstream fileLabels(filenameLabels, std::ios::in | std::ios::binary);
	if (!fileLabels)
	{
		output << "Labels bin file not found. : " << filenameLabels << std::endl;
	}

	fileImages.read((char *)&magicImages, sizeof(int));
	fileImages.read((char *)&countImages, sizeof(int));
	fileImages.read((char *)&height, sizeof(int));
	fileImages.read((char *)&width, sizeof(int));

	magicImages = ntohl(magicImages);
	countImages = ntohl(countImages);
	height = ntohl(height);
	width = ntohl(width);

	fileLabels.read((char *)&magicLabels, sizeof(int));
	fileLabels.read((char *)&countLabels, sizeof(int));
	magicLabels = ntohl(magicLabels);
	countLabels = ntohl(countLabels);

	output << "magicImages : " << magicImages << std::endl;
	output << "magicLabels : " << magicLabels << std::endl;
	output << "countImages : " << countImages << std::endl;
	output << "countLabels : " << countLabels << std::endl;
	output << "height :  " << height << std::endl;
	output << "width : " << width << std::endl;

	for (int i = 0; i < countImages; i++)
	{
		cv::Mat image(height, width, CV_8UC1);
		fileImages.read((char *)image.data, width * height);
		unsigned char label;
		fileLabels.read((char *)&label, 1);

		images.push_back(image);
		labels.push_back(label);
	}
}

void serialize_mnist(const std::vector<cv::Mat> images, const std::vector<int> labels, float **ppSerialImages, float **ppSerialLabels)
{
	int trainCount = images.size();

	int width = images[0].cols;
	int height = images[0].rows;
	int size = width * height;

	float *serialImages = new float[width * height * trainCount];
	float *serialLabels = new float[trainCount];

	int iData = 0;
	int iLabel = 0;
	for (int i = 0; i < trainCount; i++)
	{
		const cv::Mat &image = images[i];
		const unsigned char *data = image.data;
		int label = labels[i];

		for (int k = 0; k < size; k++)
		{
			serialImages[iData++] = (float)data[k];
		}
		serialLabels[iLabel++] = (float)label;
	}

	printf("::%d, %d\n", iData, iLabel);

	*ppSerialImages = serialImages;
	*ppSerialLabels = serialLabels;
}

int main()
{
	float *data = new float[64 * 1 * 1 * 2 * 400];
	float *label = new float[64 * 1 * 1 * 1 * 400];

	float testab[] = {0, 0, 0, 1, 1, 0, 1, 1};
	float testc[] = {0, 2, 1, 3};

	for (int i = 0; i < 64 * 1 * 1 * 400; ++i)
	{
		int a = rand() % 2;
		int b = rand() % 2;
		int c = a ^ b;
		data[i * 2 + 0] = a;
		data[i * 2 + 1] = b;
		label[i] = a + b * 2;
	}

	std::vector<cv::Mat> trainImages;
	std::vector<int> trainLabels;
	load_mnist("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte", trainImages, trainLabels);

	std::vector<cv::Mat> testImages;
	std::vector<int> testLabels;
	load_mnist("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte", testImages, testLabels);

	float *serialTrainImages;
	float *serialTrainLabels;

	float *serialTestImages;
	float *serialTestLabels;

	serialize_mnist(trainImages, trainLabels, &serialTrainImages, &serialTrainLabels);
	serialize_mnist(testImages, testLabels, &serialTestImages, &serialTestLabels);

	int trainCount = trainImages.size() == trainLabels.size() ? trainImages.size() : 0;

	if (trainCount == 0)
	{
		return 0;
	}

	int testCount = testImages.size() == testLabels.size() ? testImages.size() : 0;

	if (testCount == 0)
	{
		return 0;
	}

	int width = trainImages[0].cols;
	int height = trainImages[0].rows;
	int size = width * height;

	caffe::SolverParameter solver_param;
	caffe::ReadSolverParamsFromTextFileOrDie("../alex_solver.prototxt", &solver_param);

	boost::shared_ptr<caffe::Solver<float>> solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

	caffe::MemoryDataLayer<float> *dataLayer_trainnet =
		(caffe::MemoryDataLayer<float> *)(solver->net()->layer_by_name("train_data").get());
	dataLayer_trainnet->Reset(serialTrainImages, serialTrainLabels, trainCount);
	//dataLayer_trainnet->Reset(data, label, 25600);

	const auto &layers = solver->test_nets()[0];

	caffe::MemoryDataLayer<float> *dataLayer_testnet =
		(caffe::MemoryDataLayer<float> *)(solver->test_nets()[0]->layer_by_name("test_data").get());
	dataLayer_testnet->Reset(serialTestImages, serialTestLabels, testCount);
	//dataLayer_testnet->Reset(testab, testc, 4);

	

	solver->Solve();

	//test--------------------------------------

	
    boost::shared_ptr<caffe::Net<float> > testnet;

    testnet.reset(new caffe::Net<float>("../xor_model.prototxt", caffe::TEST));
    testnet->CopyTrainedLayersFrom("XOR_iter_50000.caffemodel");

    testnet->ShareTrainedLayersWith(solver->net().get());

    caffe::MemoryDataLayer<float> *dataLayer_confnet = (caffe::MemoryDataLayer<float> *) (testnet->layer_by_name("test_data").get());

    dataLayer_confnet->Reset(testab, testc, 4);

    testnet->Forward();

    boost::shared_ptr<caffe::Blob<float> > output_layer = testnet->blob_by_name("output");

    const float* begin = output_layer->cpu_data();
    const float* end = begin + 4;
    
    std::vector<float> result(begin, end);

    for(int i = 0; i< result.size(); ++i)
    {
    	printf("input: %d xor %d,  truth: %f result by nn: %f\n", (int)testab[i*2 + 0], (int)testab[i*2+1], testc[i], result[i]);
    }

	return 0;
}