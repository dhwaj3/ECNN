package CNN;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import CNN.Layer.Size;
import dataset.Dataset;
import dataset.Dataset.Record;
import util.ConcurenceRunner.TaskManager;
import util.Log;
import util.Util;
import util.Util.Operator;

public class CNN implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 337920299147929932L;
	private static double ALPHA = 0.85;
	protected static final double LAMBDA = 0;
	
        //Various layers of the network
	private List<Layer> layers;
	
        // Layers
	private int layerNum;

	// Batch update size
	private int batchSize;
	// The divisor operator divides each element of the matrix by a value
	private Operator divide_batchSize;

	// The multiplier operator multiplies each element of the matrix by the alpha valueֵ
	private Operator multiply_alpha;

	// Multiplier operator, multiplying each element of the matrix by the 1-labmda * alpha value
	private Operator multiply_lambda;

	/**
	 * Initialize the network
	 * 
	 * @param layerBuilder
	 *  Network layer
         * 
	 * @param inputMapSize
	 *  Enter the size of the map
	 * @param classNum
	 *            The number of categories requires the dataset to convert the class label to a value of 0-classNum-1
	 */
	public CNN(LayerBuilder layerBuilder, final int batchSize) {
		layers = layerBuilder.mLayers;
		layerNum = layers.size();
		this.batchSize = batchSize;
		setup(batchSize);
		initPerator();
	}

	/**
	 * Initialization operator
	 */
	private void initPerator() {
		divide_batchSize = new Operator() {

			private static final long serialVersionUID = 7424011281732651055L;

			@Override
			public double process(double value) {
				return value / batchSize;
			}

		};
		multiply_alpha = new Operator() {

			private static final long serialVersionUID = 5761368499808006552L;

			@Override
			public double process(double value) {

				return value * ALPHA;
			}

		};
		multiply_lambda = new Operator() {

			private static final long serialVersionUID = 4499087728362870577L;

			@Override
			public double process(double value) {

				return value * (1 - LAMBDA * ALPHA);
			}

		};
	}

	/**
	 * Train the network on the training set
	 * 
	 * @param trainset
	 * @param repeat
	 *         The number of iterations
	 */
	public void train(Dataset trainset, int repeat) {
		// Monitor stop button
		new Lisenter().start();
		for (int t = 0; t < repeat && !stopTrain.get(); t++) {
			int epochsNum = trainset.size() / batchSize;
			if (trainset.size() % batchSize != 0)
				epochsNum++;// Extract once, round up
			Log.i("");
			Log.i(t + "th iter epochsNum:" + epochsNum);
			int right = 0;
			int count = 0;
			for (int i = 0; i < epochsNum; i++) {
				int[] randPerm = Util.randomPerm(trainset.size(), batchSize);
				Layer.prepareForNewBatch();

				for (int index : randPerm) {
					boolean isRight = train(trainset.getRecord(index));
					if (isRight)
						right++;
					count++;
					Layer.prepareForNewRecord();
				}

				// After finishing a batch update weight
				updateParas();
				if (i % 50 == 0) {
					System.out.print("..");
					if (i + 50 > epochsNum)
						System.out.println();
				}
			}
			double p = 1.0 * right / count;
			if (t % 10 == 1 && p > 0.96) {//Adjust the quasi-learning rate dynamically
				ALPHA = 0.001 + ALPHA * 0.9;
				Log.i("Set alpha = " + ALPHA);
			}
			Log.i("precision " + right + "/" + count + "=" + p);
		}
	}

	private static AtomicBoolean stopTrain;

	static class Lisenter extends Thread {
		Lisenter() {
			setDaemon(true);
			stopTrain = new AtomicBoolean(false);
		}

		@Override
		public void run() {
			System.out.println("Input & to stop train.");
			while (true) {
				try {
					int a = System.in.read();
					if (a == '&') {
						stopTrain.compareAndSet(false, true);
						break;
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			System.out.println("Lisenter stop");
		}

	}

	/**
	 * Test Data
         * 
	 * @param trainset
	 * @return
	 */
	public double test(Dataset trainset) {
		Layer.prepareForNewBatch();
		Iterator<Record> iter = trainset.iter();
		int right = 0;
		while (iter.hasNext()) {
			Record record = iter.next();
			forward(record);
			Layer outputLayer = layers.get(layerNum - 1);
			int mapNum = outputLayer.getOutMapNum();		
			double[] out = new double[mapNum];
			for (int m = 0; m < mapNum; m++) {
				double[][] outmap = outputLayer.getMap(m);
				out[m] = outmap[0][0];
			}
			if (record.getLable().intValue() == Util.getMaxIndex(out))
				right++;		
		}
		double p = 1.0 * right / trainset.size();
		Log.i("precision", p + "");
		return p;
	}

	/**
	 * forecast result
	 * 
	 * @param testset
	 * @param fileName
	 */
	public void predict(Dataset testset, String fileName) {
		Log.i("begin predict");
		try {
			int max = layers.get(layerNum - 1).getClassNum();
			PrintWriter writer = new PrintWriter(new File(fileName));
			Layer.prepareForNewBatch();
			Iterator<Record> iter = testset.iter();
			while (iter.hasNext()) {
				Record record = iter.next();
				forward(record);
				Layer outputLayer = layers.get(layerNum - 1);

				int mapNum = outputLayer.getOutMapNum();
				double[] out = new double[mapNum];
				for (int m = 0; m < mapNum; m++) {
					double[][] outmap = outputLayer.getMap(m);
					out[m] = outmap[0][0];
				}
				// int lable =
				// Util.binaryArray2int(out);
				int lable = Util.getMaxIndex(out);
				// if (lable >= max)
				// lable = lable - (1 << (out.length -
				// 1));
				writer.write(lable + "\n");
			}
			writer.flush();
			writer.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		Log.i("end predict");
	}

	private boolean isSame(double[] output, double[] target) {
		boolean r = true;
		for (int i = 0; i < output.length; i++)
			if (Math.abs(output[i] - target[i]) > 0.5) {
				r = false;
				break;
			}

		return r;
	}

	/**
	 *  Training a record, at the same time return to predict the correct current record
	 * 
	 * @param record
	 * @return
	 */
	private boolean train(Record record) {
		forward(record);
		boolean result = backPropagation(record);
		return result;
		// System.exit(0);
	}

	/*
	 * Reverse transmission
	 */
	private boolean backPropagation(Record record) {
		boolean result = setOutLayerErrors(record);
		setHiddenLayerErrors();
		return result;
	}

	/**
	 * Update parameters
	 */
	private void updateParas() {
		for (int l = 1; l < layerNum; l++) {
			Layer layer = layers.get(l);
			Layer lastLayer = layers.get(l - 1);
			switch (layer.getType()) {
			case conv:
			case output:
				updateKernels(layer, lastLayer);
				updateBias(layer, lastLayer);
				break;
			default:
				break;
			}
		}
	}

	/**
	 *  Update offset
	 * 
	 * @param layer
	 * @param lastLayer
	 */
	private void updateBias(final Layer layer, Layer lastLayer) {
		final double[][][][] errors = layer.getErrors();
		int mapNum = layer.getOutMapNum();

		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					double[][] error = Util.sum(errors, j);
					// Update offset
					double deltaBias = Util.sum(error) / batchSize;
					double bias = layer.getBias(j) + ALPHA * deltaBias;
					layer.setBias(j, bias);
				}
			}
		}.start();

	}

	/**
	 * Update the layer layer convolution kernel (weight) and offset
	 * 
	 * @param layer
	 *           The current level
	 * @param lastLayer
	 *            The previous floor
	 */
	private void updateKernels(final Layer layer, final Layer lastLayer) {
		int mapNum = layer.getOutMapNum();
		final int lastMapNum = lastLayer.getOutMapNum();
		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					for (int i = 0; i < lastMapNum; i++) {
						// Sums each record delta for the batch
						double[][] deltaKernel = null;
						for (int r = 0; r < batchSize; r++) {
							double[][] error = layer.getError(r, j);
							if (deltaKernel == null)
								deltaKernel = Util.convnValid(
										lastLayer.getMap(r, i), error);
							else {// Summing up
								deltaKernel = Util.matrixOp(Util.convnValid(
										lastLayer.getMap(r, i), error),
										deltaKernel, null, null, Util.plus);
							}
						}

						// Divide by batchSize
						deltaKernel = Util.matrixOp(deltaKernel,
								divide_batchSize);
						// Update the convolution kernel
						double[][] kernel = layer.getKernel(i, j);
						deltaKernel = Util.matrixOp(kernel, deltaKernel,
								multiply_lambda, multiply_alpha, Util.plus);
						layer.setKernel(i, j, deltaKernel);
					}
				}

			}
		}.start();

	}

	/**
	 * Set the median will be the residual layers
	 */
	private void setHiddenLayerErrors() {
		for (int l = layerNum - 2; l > 0; l--) {
			Layer layer = layers.get(l);
			Layer nextLayer = layers.get(l + 1);
			switch (layer.getType()) {
			case samp:
				setSampErrors(layer, nextLayer);
				break;
			case conv:
				setConvErrors(layer, nextLayer);
				break;
			default:// ֻOnly the sampling layer and the convolution layer need to deal with the residuals, 
                                //  the input layer has no residuals and the output layer has been processed
				break;
			}
		}
	}

	/**
	 * Set the sampling layer residuals
	 * 
	 * @param layer
	 * @param nextLayer
	 */
	private void setSampErrors(final Layer layer, final Layer nextLayer) {
		int mapNum = layer.getOutMapNum();
		final int nextMapNum = nextLayer.getOutMapNum();
		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int i = start; i < end; i++) {
					double[][] sum = null;// Summarize each convolution
					for (int j = 0; j < nextMapNum; j++) {
						double[][] nextError = nextLayer.getError(j);
						double[][] kernel = nextLayer.getKernel(i, j);
						// Rotate the convolution kernel by 180 degrees and then convolve in full mode
						if (sum == null)
							sum = Util
									.convnFull(nextError, Util.rot180(kernel));
						else
							sum = Util.matrixOp(
									Util.convnFull(nextError,
											Util.rot180(kernel)), sum, null,
									null, Util.plus);
					}
					layer.setError(i, sum);
				}
			}

		}.start();

	}

	/**
	 * Set the convolutional layer residuals
	 * 
	 * @param layer
	 * @param nextLayer
	 */
	private void setConvErrors(final Layer layer, final Layer nextLayer) {
		// The next layer of the convolutional layer is the sampling layer, that is, the two layers have the same number of maps, and one map connects only with one map of the first layer,
               // So just spread the next layer of residual kronecker to dot product
		int mapNum = layer.getOutMapNum();
		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int m = start; m < end; m++) {
					Size scale = nextLayer.getScaleSize();
					double[][] nextError = nextLayer.getError(m);
					double[][] map = layer.getMap(m);
					//Matrix multiplication, but 1-value operation on each element value of the second matrix
					double[][] outMatrix = Util.matrixOp(map,
							Util.cloneMatrix(map), null, Util.one_value,
							Util.multiply);
					outMatrix = Util.matrixOp(outMatrix,
							Util.kronecker(nextError, scale), null, null,
							Util.multiply);
					layer.setError(m, outMatrix);
				}

			}

		}.start();

	}

	/**
	 * Set the output layer of the residual value, the number of output neurons fewer units, not to consider multi-threading
	 * 
	 * @param record
	 * @return
	 */
	private boolean setOutLayerErrors(Record record) {

		Layer outputLayer = layers.get(layerNum - 1);
		int mapNum = outputLayer.getOutMapNum();
		// double[] target =
		// record.getDoubleEncodeTarget(mapNum);
		// double[] outmaps = new double[mapNum];
		// for (int m = 0; m < mapNum; m++) {
		// double[][] outmap = outputLayer.getMap(m);
		// double output = outmap[0][0];
		// outmaps[m] = output;
		// double errors = output * (1 - output) *
		// (target[m] - output);
		// outputLayer.setError(m, 0, 0, errors);
		// }
		// // correct
		// if (isSame(outmaps, target))
		// return true;
		// return false;

		double[] target = new double[mapNum];
		double[] outmaps = new double[mapNum];
		for (int m = 0; m < mapNum; m++) {
			double[][] outmap = outputLayer.getMap(m);
			outmaps[m] = outmap[0][0];

		}
		int lable = record.getLable().intValue();
		target[lable] = 1;
		// Log.i(record.getLable() + "outmaps:" +
		// Util.fomart(outmaps)
		// + Arrays.toString(target));
		for (int m = 0; m < mapNum; m++) {
			outputLayer.setError(m, 0, 0, outmaps[m] * (1 - outmaps[m])
					* (target[m] - outmaps[m]));
		}
		return lable == Util.getMaxIndex(outmaps);
	}

	/**
	 * Forward calculation of a record
	 * 
	 * @param record
	 */
	private void forward(Record record) {
		// Set the map of the input layer
		setInLayerOutput(record);
		for (int l = 1; l < layers.size(); l++) {
			Layer layer = layers.get(l);
			Layer lastLayer = layers.get(l - 1);
			switch (layer.getType()) {
			case conv:// Compute the output of the convolution layer
				setConvOutput(layer, lastLayer);
				break;
			case samp:// Calculate the output of the sampling layer
				setSampOutput(layer, lastLayer);
				break;
			case output:// Calculate the output of the output layer, the output layer is a special convolution layer
				setConvOutput(layer, lastLayer);
				break;
			default:
				break;
			}
		}
	}

	/**
	 * Set the input layers output value based on the recorded value
	 * 
	 * @param record
	 */
	private void setInLayerOutput(Record record) {
		final Layer inputLayer = layers.get(0);
		final Size mapSize = inputLayer.getMapSize();
		final double[] attr = record.getAttrs();
		if (attr.length != mapSize.x * mapSize.y)
			throw new RuntimeException("The size of the data record does not match the size of the map defined!");
		for (int i = 0; i < mapSize.x; i++) {
			for (int j = 0; j < mapSize.y; j++) {
				// A one-dimensional vector of recording properties is made into a two-dimensional matrix
				inputLayer.setMapValue(0, i, j, attr[mapSize.x * i + j]);
			}
		}
	}

	/*
	 * Compute the output of the convolutional layer, each thread is responsible for part of the map
	 */
	private void setConvOutput(final Layer layer, final Layer lastLayer) {
		int mapNum = layer.getOutMapNum();
		final int lastMapNum = lastLayer.getOutMapNum();
		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					double[][] sum = null;// Sum the convolution of each input map
					for (int i = 0; i < lastMapNum; i++) {
						double[][] lastMap = lastLayer.getMap(i);
						double[][] kernel = layer.getKernel(i, j);
						if (sum == null)
							sum = Util.convnValid(lastMap, kernel);
						else
							sum = Util.matrixOp(
									Util.convnValid(lastMap, kernel), sum,
									null, null, Util.plus);
					}
					final double bias = layer.getBias(j);
					sum = Util.matrixOp(sum, new Operator() {
						private static final long serialVersionUID = 2469461972825890810L;

						@Override
						public double process(double value) {
							return Util.sigmod(value + bias);
						}

					});

					layer.setMapValue(j, sum);
				}
			}

		}.start();

	}

	/**
	 * Set the output value of the sampling layer. The sampling layer is the mean processing of the convolution layer
	 * 
	 * @param layer
	 * @param lastLayer
	 */
	private void setSampOutput(final Layer layer, final Layer lastLayer) {
		int lastMapNum = lastLayer.getOutMapNum();
		new TaskManager(lastMapNum) {

			@Override
			public void process(int start, int end) {
				for (int i = start; i < end; i++) {
					double[][] lastMap = lastLayer.getMap(i);
					Size scaleSize = layer.getScaleSize();
					// The scaleSize area for averaging
					double[][] sampMatrix = Util
							.scaleMatrix(lastMap, scaleSize);
					layer.setMapValue(i, sampMatrix);
				}
			}

		}.start();

	}

	/**
	 * Set cnn network parameters of each layer
	 * 
	 * @param batchSize
	 *            * @param classNum
	 * @param inputMapSize
	 */
	public void setup(int batchSize) {
		Layer inputLayer = layers.get(0);
		// Each layer needs to initialize the output map
		inputLayer.initOutmaps(batchSize);
		for (int i = 1; i < layers.size(); i++) {
			Layer layer = layers.get(i);
			Layer frontLayer = layers.get(i - 1);
			int frontMapNum = frontLayer.getOutMapNum();
			switch (layer.getType()) {
			case input:
				break;
			case conv:
				// Set the size of the map
				layer.setMapSize(frontLayer.getMapSize().subtract(
						layer.getKernelSize(), 1));
				//Initializes the convolution kernel with a total of frontMapNum * outMapNum convolution kernels

				layer.initKernel(frontMapNum);
				// Initialize offset, total frontMapNum * outMapNum offset
				layer.initBias(frontMapNum);
				// Each record in the batch should have a residual
				layer.initErros(batchSize);
				// Each layer needs to initialize the output map
				layer.initOutmaps(batchSize);
				break;
			case samp:
				// The number of map layers is the same as the previous map
				layer.setOutMapNum(frontMapNum);
				// The size of the sample map is the size of the previous map divided by the scale
				layer.setMapSize(frontLayer.getMapSize().divide(layer.getScaleSize()));
				// Each record in the batch should have a residual
				layer.initErros(batchSize);
				// Each layer needs to initialize the output map
				layer.initOutmaps(batchSize);
				break;
			case output:
				// Initialization weight (convolution kernel), output layer convolution kernel size of the previous map size
				layer.initOutputKerkel(frontMapNum, frontLayer.getMapSize());
				// Initialize offset, total frontMapNum * outMapNum offset
				layer.initBias(frontMapNum);
				// Each record in the batch should have a residual
				layer.initErros(batchSize);
				// Each layer needs to initialize the output map
				layer.initOutmaps(batchSize);
				break;
			}
		}
	}

	/**
	 * Constructor mode Constructs layers, requiring that the second last must be a sampling layer and not a convolution

	 * 
	 * @author jiqunpeng
	 * 
	 *        Created: 2014-7-8 下午 4:54:29
	 */
	public static class LayerBuilder {
		private List<Layer> mLayers;

		public LayerBuilder() {
			mLayers = new ArrayList<Layer>();
		}

		public LayerBuilder(Layer layer) {
			this();
			mLayers.add(layer);
		}

		public LayerBuilder addLayer(Layer layer) {
			mLayers.add(layer);
			return this;
		}
	}

	/**
	 * Serialize the save model
	 * 
	 * @param fileName
	 */
	public void saveModel(String fileName) {
		try {
			ObjectOutputStream oos = new ObjectOutputStream(
					new FileOutputStream(fileName));
			oos.writeObject(this);
			oos.flush();
			oos.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	/**
	 * 
	 *  Deserialize the import model
	 * @param fileName
	 * @return
	 */
	public static CNN loadModel(String fileName) {
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(
					fileName));
			CNN cnn = (CNN) in.readObject();
			in.close();
			return cnn;
		} catch (IOException | ClassNotFoundException e) {
			e.printStackTrace();
		}
		return null;
	}
}
