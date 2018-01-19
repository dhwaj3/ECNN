package CNN;

import java.io.Serializable;

import edu.hitsz.c102c.util.Log;
import edu.hitsz.c102c.util.Util;

/**
 * cnn����Ĳ�
 * 
 * @author jiqunpeng
 * 
 *         ����ʱ�䣺2014-7-8 ����3:58:46
 */
public class Layer implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -5747622503947497069L;
	private LayerType type;// �������
	private int outMapNum;// ���map�ĸ���
	private Size mapSize;// map�Ĵ�С
	private Size kernelSize;// ����˴�С��ֻ�о������
	private Size scaleSize;// ������С��ֻ�в�������
	private double[][][][] kernel;// ����ˣ�ֻ�о������������
	private double[] bias;// ÿ��map��Ӧһ��ƫ�ã�ֻ�о������������
	// �������batch�����map��outmaps[0][0]��ʾ��һ����¼ѵ���µ�0�����map
	private double[][][][] outmaps;
	// �в��matlab toolbox��d��Ӧ
	private double[][][][] errors;

	private static int recordInBatch = 0;// ��¼��ǰѵ������batch�ĵڼ�����¼

	private int classNum = -1;// ������

	private Layer() {

	}

	/**
	 * ׼����һ��batch��ѵ��
	 */
	public static void prepareForNewBatch() {
		recordInBatch = 0;
	}

	/**
	 * ׼����һ����¼��ѵ��
	 */
	public static void prepareForNewRecord() {
		recordInBatch++;
	}

	/**
	 * ��ʼ�������
	 * 
	 * @param mapSize
	 * @return
	 */
	public static Layer buildInputLayer(Size mapSize) {
		Layer layer = new Layer();
		layer.type = LayerType.input;
		layer.outMapNum = 1;// ������map����Ϊ1����һ��ͼ
		layer.setMapSize(mapSize);//
		return layer;
	}

	/**
	 * ��������
	 * 
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
	 * ���������
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
	 * ���������,���������������ĸ��������������Ԫ�ĸ���
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
	 * ��ȡmap�Ĵ�С
	 * 
	 * @return
	 */
	public Size getMapSize() {
		return mapSize;
	}

	/**
	 * ����map�Ĵ�С
	 * 
	 * @param mapSize
	 */
	public void setMapSize(Size mapSize) {
		this.mapSize = mapSize;
	}

	/**
	 * ��ȡ�������
	 * 
	 * @return
	 */
	public LayerType getType() {
		return type;
	}

	/**
	 * ��ȡ�����������
	 * 
	 * @return
	 */

	public int getOutMapNum() {
		return outMapNum;
	}

	/**
	 * �������map�ĸ���
	 * 
	 * @param outMapNum
	 */
	public void setOutMapNum(int outMapNum) {
		this.outMapNum = outMapNum;
	}

	/**
	 * ��ȡ����˵Ĵ�С��ֻ�о������kernelSize���������δnull
	 * 
	 * @return
	 */
	public Size getKernelSize() {
		return kernelSize;
	}

	/**
	 * ��ȡ������С��ֻ�в�������scaleSize���������δnull
	 * 
	 * @return
	 */
	public Size getScaleSize() {
		return scaleSize;
	}

	enum LayerType {
		// ���������ͣ�����㡢����㡢����㡢������
		input, output, conv, samp
	}

	/**
	 * ����˻��߲�����scale�Ĵ�С,�������Բ���.���Ͱ�ȫ�����Ժ󲻿��޸�
	 * 
	 * @author jiqunpeng
	 * 
	 *         ����ʱ�䣺2014-7-8 ����4:11:00
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
		 * ����scaleSize�õ�һ���µ�Size��Ҫ��this.x��this.
		 * y�ֱܷ�scaleSize.x��scaleSize.y����
		 * 
		 * @param scaleSize
		 * @return
		 */
		public Size divide(Size scaleSize) {
			int x = this.x / scaleSize.x;
			int y = this.y / scaleSize.y;
			if (x * scaleSize.x != this.x || y * scaleSize.y != this.y)
				throw new RuntimeException(this + "��������" + scaleSize);
			return new Size(x, y);
		}

		/**
		 * ��ȥsize��С����x��y�ֱ𸽼�һ��ֵappend
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
	 * �����ʼ�������
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
	 * �����ľ���˵Ĵ�С����һ���map��С
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
	 * ��ʼ��ƫ��
	 * 
	 * @param frontMapNum
	 */
	public void initBias(int frontMapNum) {
		this.bias = Util.randomArray(outMapNum);
	}

	/**
	 * ��ʼ�����map
	 * 
	 * @param batchSize
	 */
	public void initOutmaps(int batchSize) {
		outmaps = new double[batchSize][outMapNum][mapSize.x][mapSize.y];
	}

	/**
	 * ����mapֵ
	 * 
	 * @param mapNo
	 *            �ڼ���map
	 * @param mapX
	 *            map�ĸ�
	 * @param mapY
	 *            map�Ŀ�
	 * @param value
	 */
	public void setMapValue(int mapNo, int mapX, int mapY, double value) {
		outmaps[recordInBatch][mapNo][mapX][mapY] = value;
	}

	static int count = 0;

	/**
	 * �Ծ�����ʽ���õ�mapNo��map��ֵ
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
	 * ��ȡ��index��map���󡣴������ܿ��ǣ�û�з��ظ��ƶ��󣬶���ֱ�ӷ������ã����ö��������
	 * �����޸�outmaps�������޸������setMapValue(...)
	 * 
	 * @param index
	 * @return
	 */
	public double[][] getMap(int index) {
		return outmaps[recordInBatch][index];
	}

	/**
	 * ��ȡǰһ���i��map����ǰ���j��map�ľ����
	 * 
	 * @param i
	 *            ��һ���map�±�
	 * @param j
	 *            ��ǰ���map�±�
	 * @return
	 */
	public double[][] getKernel(int i, int j) {
		return kernel[i][j];
	}

	/**
	 * ���òв�ֵ
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
	 * ��map�������ʽ���òв�ֵ
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
	 * ��ȡ��mapNo��map�Ĳв�.û�з��ظ��ƶ��󣬶���ֱ�ӷ������ã����ö��������
	 * �����޸�errors�������޸������setError(...)
	 * 
	 * @param mapNo
	 * @return
	 */
	public double[][] getError(int mapNo) {
		return errors[recordInBatch][mapNo];
	}

	/**
	 * ��ȡ����(ÿ����¼��ÿ��map)�Ĳв�
	 * 
	 * @return
	 */
	public double[][][][] getErrors() {
		return errors;
	}

	/**
	 * ��ʼ���в�����
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
	 * ��ȡ��mapNo��
	 * 
	 * @param mapNo
	 * @return
	 */
	public double getBias(int mapNo) {
		return bias[mapNo];
	}

	/**
	 * ���õ�mapNo��map��ƫ��ֵ
	 * 
	 * @param mapNo
	 * @param value
	 */
	public void setBias(int mapNo, double value) {
		bias[mapNo] = value;
	}

	/**
	 * ��ȡbatch����map����
	 * 
	 * @return
	 */

	public double[][][][] getMaps() {
		return outmaps;
	}

	/**
	 * ��ȡ��recordId��¼�µ�mapNo�Ĳв�
	 * 
	 * @param recordId
	 * @param mapNo
	 * @return
	 */
	public double[][] getError(int recordId, int mapNo) {
		return errors[recordId][mapNo];
	}

	/**
	 * ��ȡ��recordId��¼�µ�mapNo�����map
	 * 
	 * @param recordId
	 * @param mapNo
	 * @return
	 */
	public double[][] getMap(int recordId, int mapNo) {
		return outmaps[recordId][mapNo];
	}

	/**
	 * ��ȡ������
	 * 
	 * @return
	 */
	public int getClassNum() {
		return classNum;
	}

	/**
	 * ��ȡ���еľ����
	 * 
	 * @return
	 */
	public double[][][][] getKernel() {
		return kernel;
	}

}
