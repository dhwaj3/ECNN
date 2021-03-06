package CNN;

import CNN.CNN.LayerBuilder;
import CNN.Layer.Size;
import dataset.Dataset;
import util.ConcurenceRunner;
import util.TimedTest;
import util.TimedTest.TestTask;

public class RunCNN {

	public static void runCnn() {
		//Create a convolutional neural network
		LayerBuilder builder = new LayerBuilder();
		builder.addLayer(Layer.buildInputLayer(new Size(28, 28)));
		builder.addLayer(Layer.buildConvLayer(6, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildConvLayer(12, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildOutputLayer(10));
		CNN cnn = new CNN(builder, 50);
		
		//Import the data set
		String fileName = "dataset/train.format";
		Dataset dataset = Dataset.load(fileName, ",", 784);
		cnn.train(dataset, 3);//
		String modelName = "model/model.cnn";
		cnn.saveModel(modelName);		
		dataset.clear();
		dataset = null;
		
		//prediction
		// CNN cnn = CNN.loadModel(modelName);	
		Dataset testset = Dataset.load("dataset/test.format", ",", -1);
		cnn.predict(testset, "dataset/test.predict");
	}

	public static void main(String[] args) {

		new TimedTest(new TestTask() {

			@Override
			public void process() {
				runCnn();
			}
		}, 1).test();
		ConcurenceRunner.stop();

	}

}
