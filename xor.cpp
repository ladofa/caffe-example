#include <caffe/caffe.hpp>
#include <memory>
#include "caffe/layers/memory_data_layer.hpp"

int main()
{
	float *data = new float[64*1*1*3*400];
	float *label = new float[64*1*1*1*400];


	for(int i = 0; i<64*1*1*400; ++i)
	{
		int a = rand() % 2;
        int b = rand() % 2;
        int c = a ^ b;
		data[i*2 + 0] = a;
		data[i*2 + 1] = b;
		label[i] = c;
	}

    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie("./solver.prototxt", &solver_param);

    boost::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
    caffe::MemoryDataLayer<float> *dataLayer_trainnet = (caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("inputdata").get());
    caffe::MemoryDataLayer<float> *dataLayer_testnet_ = (caffe::MemoryDataLayer<float> *) (solver->test_nets()[0]->layer_by_name("test_inputdata").get());

    float testab[] = {0, 0, 0, 1, 1, 0, 1, 1};
    float testc[] = {0, 1, 1, 0};

    dataLayer_testnet_->Reset(testab, testc, 4);

    dataLayer_trainnet->Reset(data, label, 25600);

    solver->Solve();

    boost::shared_ptr<caffe::Net<float> > testnet;

    testnet.reset(new caffe::Net<float>("./model.prototxt", caffe::TEST));
    //testnet->CopyTrainedLayersFrom("XOR_iter_5000000.caffemodel");

    testnet->ShareTrainedLayersWith(solver->net().get());

    caffe::MemoryDataLayer<float> *dataLayer_testnet = (caffe::MemoryDataLayer<float> *) (testnet->layer_by_name("test_inputdata").get());

    dataLayer_testnet->Reset(testab, testc, 4);

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