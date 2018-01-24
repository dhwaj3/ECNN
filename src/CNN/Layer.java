package CNN;

import java.io.Serializable;

import util.Log;
import util.Util;

/**
 * CNN network layer
 * 
 * @author Dhwaj verma
 * 
 *        
 */
public class Layer implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -5747622503947497069L;
	private LayerType type;// The type of layer
	private int outMapNum;//The number of map output
	private Size mapSize;// The size of the map
	private Size kernelSize;// Convolution kernel size, only the convolution layer
	private Size scaleSize;// Sampling size, only the sampling layer
	private double[][][][] kernel;// Convolution kernel, only convolution layer and output layer
	private double[] bias;// Each map corresponds to a bias, only the convolutional layer and the output layer
	// Save the output of each batch map, outmaps [0] [0] said the first record training 0th output map
	private double[][][][] outmaps;
	// Residual, and matlab toolbox d corresponding
	private double[][][][] errors;

	private static int recordInBatch = 0;// Record the current training is batch of the first few records

	private int classNum = -1;// Number of categories


	private Layer() {

	}

	/**
	 * Prepare for the next batch of training
	 */
	public static void prepareForNewBatch() {
		recordInBatch = 0;
	}

	/**
	 * Prepare for the next record of training
	 */
	public static void prepareForNewRecord() {
		recordInBatch++;
	}

	/**
	 * Initialize the input layer
	 * 
	 * @param mapSize
	 * @return
	 */
	public static Layer buildInputLayer(Size mapSize) {
		Layer layer = new Layer();
		layer.type = LayerType.input;
		layer.outMapNum = 1;// The number of map input layer 1, that is, a map
		layer.setMapSize(mapSize);//
		return layer;
	}

	/**
	 * Build a convolution layer
	 * @return
	 */
	public static Layer buildConvLayer(int outMapNum, Size kernelSize) {
		Layer layer = new Layer();
		layer.type = LayerType.conv;
		layer.outMapNum = outMapNum;
		layer.kernelSize = kernelSize;
		return layer;
	}

	/**
	 * Structure the sampling layer
	 * 
	 * @param scaleSize
	 * @return
	 */
	public static Layer buildSampLayer(Size scaleSize) {
		Layer layer = new Layer();
		layer.type = LayerType.samp;
		layer.scaleSize = scaleSize;
		return layer;
	}

	/**
	 * Structure output layer, the number of categories,
         * according to the number of categories to determine the number of output units
	 * 
	 * @return
	 */
	public static Layer buildOutputLayer(int classNum) {
		Layer layer = new Layer();
		layer.classNum = classNum;
		layer.type = LayerType.output;
		layer.mapSize = new Size(1, 1);
		layer.outMapNum = classNum;
		// int outMapNum = 1;
		// while ((1 << outMapNum) < classNum)
		// outMapNum += 1;
		// layer.outMapNum = outMapNum;
		Log.i("outMapNum:" + layer.outMapNum);
		return layer;
	}

	/**
	 * Get the size of the map
	 * 
	 * @return
	 */
	public Size getMapSize() {
		return mapSize;
	}

	/**
	 * 
	 * Get the size of the map

	 * @param mapSize
	 */
	public void setMapSize(Size mapSize) {
		this.mapSize = mapSize;
	}

	/**
	 * Get the type of layer
	 * 
	 * @return
	 */
	public LayerType getType() {
		return type;
	}

	/**
	 * Get the number of output vectors
	 * 
	 * @return
	 */

	public int getOutMapNum() {
		return outMapNum;
	}

	/**
	 * Set the number of output map
	 * 
	 * @param outMapNum
	 */
	public void setOutMapNum(int outMapNum) {
		this.outMapNum = outMapNum;
	}

	/**
	 * Get the size of the convolution kernel, only convolutional kernelSize, other layers are not null
	 * 
	 * @return
	 */
	public Size getKernelSize() {
		return kernelSize;
	}

	/**
	 * Get the sample size, only the sample layer scaleSize, other layers are not null
	 * 
	 * @return
	 */
	public Size getScaleSize() {
		return scaleSize;
	}

	enum LayerType {
		// Network layer types: input layer, output layer, convolution layer, sampling layer
		input, output, conv, samp
	}

	/**
	 * Convolution kernel or sample layer scale size, length and width may vary. 
         * Type safety, can not be modified after setting
	 * 
	 * @author jiqunpeng
	 * 
	 *         
	 */
	public static class Size implements Serializable {

		private static final long serialVersionUID = -209157832162004118L;
		public final int x;
		public final int y;

		public Size(int x, int y) {
			this.x = x;
			this.y = y;
		}

		public String toString() {
			StringBuilder s = new StringBuilder("Size(").append(" x = ")
					.append(x).append(" y= ").append(y).append(")");
			return s.toString();
		}

		/**
		 * Divide scaleSize to get a new Size, this.x, this.
                 * y can be divided by scaleSize.x, scaleSize.y respectively
		 * 
		 * @param scaleSize
		 * @return
		 */
		public Size divide(Size scaleSize) {
			int x = this.x / scaleSize.x;
			int y = this.y / scaleSize.y;
			if (x * scaleSize.x != this.x || y * scaleSize.y != this.y)
				throw new RuntimeException(this + "不能整除" + scaleSize);
			return new Size(x, y);
		}

		/**
		 * Subtract size and append a value append to x and y, respectively
		 * 
		 * @param size
		 * @param append
		 * @return
		 */
		public Size subtract(Size size, int append) {
			int x = this.x - size.x + append;
			int y = this.y - size.y + append;
			return new Size(x, y);
		}
	}

	/**
	 * Convolution kernels are initialized randomly
	 * 
	 * @param frontMapNum
	 */
	public void initKernel(int frontMapNum) {
//		int fan_out = getOutMapNum() * kernelSize.x * kernelSize.y;
//		int fan_in = frontMapNum * kernelSize.x * kernelSize.y;
//		double factor = 2 * Math.sqrt(6 / (fan_in + fan_out));
		this.kernel = new double[frontMapNum][outMapNum][kernelSize.x][kernelSize.y];
		for (int i = 0; i < frontMapNum; i++)
			for (int j = 0; j < outMapNum; j++)
				kernel[i][j] = Util.randomMatrix(kernelSize.x, kernelSize.y,true);
	}

	/**
	 * The size of the convolution kernel of the output layer is the map size of the previous layer
	 * 
	 * @param frontMapNum
	 * @param size
	 */
	public void initOutputKerkel(int frontMapNum, Size size) {
		kernelSize = size;
//		int fan_out = getOutMapNum() * kernelSize.x * kernelSize.y;
//		int fan_in = frontMapNum * kernelSize.x * kernelSize.y;
//		double factor = 2 * Math.sqrt(6 / (fan_in + fan_out));
		this.kernel = new double[frontMapNum][outMapNum][kernelSize.x][kernelSize.y];
		for (int i = 0; i < frontMapNum; i++)
			for (int j = 0; j < outMapNum; j++)
				kernel[i][j] = Util.randomMatrix(kernelSize.x, kernelSize.y,false);
	}

	/**
	 * Initialize the offset
	 * 
	 * @param frontMapNum
	 */
	public void initBias(int frontMapNum) {
		this.bias = Util.randomArray(outMapNum);
	}

	/**
	 * Initialize the output map
	 * 
	 * @param batchSize
	 */
	public void initOutmaps(int batchSize) {
		outmaps = new double[batchSize][outMapNum][mapSize.x][mapSize.y];
	}

	/**
	 * Set the map value
	 * 
	 * @param mapNo
	 *            The first few map
	 * @param mapX
	 *            map high
	 * @param mapY
	 *           map wide
	 * @param value
	 */
	public void setMapValue(int mapNo, int mapX, int mapY, double value) {
		outmaps[recordInBatch][mapNo][mapX][mapY] = value;
	}

	static int count = 0;

	/**
	 * Set the value of mapNo map in matrix form
	 * 
	 * @param mapNo
	 * @param outMatrix
	 */
	public void setMapValue(int mapNo, double[][] outMatrix) {
		// Log.i(type.toString());
		// Util.printMatrix(outMatrix);
		outmaps[recordInBatch][mapNo] = outMatrix;
	}

	/**
	 * Get the index map matrix. In performance considerations, 
         * did not return a copy of the object, 
         * but directly return the reference, the call side please be careful,
         * Avoid modifying out maps, please call setMapValue (...)
	 * 
	 * @param index
	 * @return
	 */
	public double[][] getMap(int index) {
		return outmaps[recordInBatch][index];
	}

	/**
	 * Get the convolution kernel of the i th map of the previous layer to the j th map of the current layer
	 * 
	 * @param i
	 *            The next level of the map subscript
	 * @param j
	 *            The current level of the map subscript
	 * @return
	 */
	public double[][] getKernel(int i, int j) {
		return kernel[i][j];
	}

	/**
	 * Set the residual value
	 * 
	 * @param mapNo
	 * @param mapX
	 * @param mapY
	 * @param value
	 */
	public void setError(int mapNo, int mapX, int mapY, double value) {
		errors[recordInBatch][mapNo][mapX][mapY] = value;
	}

	/**
	 * Set the residual value as a map matrix block
	 * 
	 * @param mapNo
	 * @param matrix
	 */
	public void setError(int mapNo, double[][] matrix) {
		// Log.i(type.toString());
		// Util.printMatrix(matrix);
		errors[recordInBatch][mapNo] = matrix;
	}

	/**
	 * Get the mapNo a map of the residual.Do not return a copy of the object, but directly return the reference, the call side please be careful,
         * Avoid modifying errors, if you need to modify setError (...)
	 * 
	 * @param mapNo
	 * @return
	 */
	public double[][] getError(int mapNo) {
		return errors[recordInBatch][mapNo];
	}

	/**
	 * Gets all the residuals (per record and per map)
	 * 
	 * @return
	 */
	public double[][][][] getErrors() {
		return errors;
	}

	/**
	 * Initialize the residual array
	 * 
	 * @param batchSize
	 */
	public void initErros(int batchSize) {
		errors = new double[batchSize][outMapNum][mapSize.x][mapSize.y];
	}

	/**
	 * 
	 * @param lastMapNo
	 * @param mapNo
	 * @param kernel
	 */
	public void setKernel(int lastMapNo, int mapNo, double[][] kernel) {
		this.kernel[lastMapNo][mapNo] = kernel;
	}

	/**
	 * Get the first mapNo
	 * 
	 * @param mapNo
	 * @return
	 */
	public double getBias(int mapNo) {
		return bias[mapNo];
	}

	/**
	 * Get the first mapNo
	 * 
	 * @param mapNo
	 * @param value
	 */
	public void setBias(int mapNo, double value) {
		bias[mapNo] = value;
	}

	/**
	 * Get batch of map matrices
	 * 
	 * @return
	 */

	public double[][][][] getMaps() {
		return outmaps;
	}

	/**
	 * Get the first recordId record the first mapNo residual
	 * 
	 * @param recordId
	 * @param mapNo
	 * @return
	 */
	public double[][] getError(int recordId, int mapNo) {
		return errors[recordId][mapNo];
	}

	/**
	 * Get the first recordId record mapNo output map
	 * 
	 * @param recordId
	 * @param mapNo
	 * @return
	 */
	public double[][] getMap(int recordId, int mapNo) {
		return outmaps[recordId][mapNo];
	}

	/**
	 * Get the number of categories
	 * 
	 * @return
	 */
	public int getClassNum() {
		return classNum;
	}

	/**
	 * Get all the convolution kernels
	 * 
	 * @return
	 */
	public double[][][][] getKernel() {
		return kernel;
	}

}