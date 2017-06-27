package neuralNetworks;

public class Perceptron {

	public static final int[][][] andData = {{{0,0}, {0}},
											 {{0,1},{0}},
											 {{1,0},{0}},
											 {{1,1},{1}}};
	
	public static final double TAXA_APRENDIZADO = 0.05;
	public static final double[] PESOS_INICIAIS = {Math.random(), Math.random()};
	public double calcularSomaPesos(int[] data, double[] pesos) {
		double somaPesos = 0;
		for (int x = 0; x < data.length; x++) somaPesos += data[x] * pesos[x];
		return somaPesos;
	
	}
	
	public int aplicarFuncaoAtivacao(double pesoSomado) {
		int resultado = 0;
		if (pesoSomado > 1) resultado = 1;
		return resultado;
	}
	public double[] ajustarPesos(int[] data, double[] pesos, double erro){
		double[] pesosAjustados = new double[pesos.length];
		for (int x = 0; x < pesos.length; x++) pesosAjustados[x] = TAXA_APRENDIZADO * erro * data[x] + pesos[x];
		return pesosAjustados;
	}

}
