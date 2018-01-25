package util;

import java.util.Locale;

import util.TimedTest.TestTask;

/**
 *The test element accesses the array directly and accesses the array through the function, concluding that the formal function access does not slow down
 * 
 * 
 */
public class TestArray {
	double[][] data;

	public TestArray(int m, int n) {
		data = new double[m][n];
	}

	public void set(int x, int y, double value) {
		data[x][y] = value;
	}

	private void useOrigin() {
		for (int i = 0; i < data.length; i++)
			for (int j = 0; j < data[0].length; j++)
				data[i][j] = i * j;
	}

	private void useFunc() {
		for (int i = 0; i < data.length; i++)
			for (int j = 0; j < data[0].length; j++)
				set(i, j, i * j);
	}

	public static void main(String[] args) {
		String a = "aAdfa Peng _";
		System.out.println(a.toUpperCase(Locale.CHINA));
		double[][] d = new double[3][];
//		d[0] = new double[] { 1,2,3 };
//		d[1] = new double[] { 3,4,5,6 };
		System.out.println(d[1][3]);
		final TestArray t = new TestArray(10000, 1000);
		new TimedTest(new TestTask() {

			@Override
			public void process() {
				t.useFunc();
			}
		}, 1).test();
	
		
	}
}