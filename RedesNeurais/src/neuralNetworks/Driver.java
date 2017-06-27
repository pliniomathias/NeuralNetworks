package neuralNetworks;

public class Driver {
	
	public static void main(String[] args){
		int [][][] data = Perceptron.andData;
		double[] pesos = Perceptron.PESOS_INICIAIS;
		Driver driver = new Driver();
		Perceptron perceptron = new Perceptron();
		int epochNumber = 0;
		boolean indicadorErro = true;
		double erro = 0;
		double[] pesosAjustados = null;
		while(indicadorErro) {
			driver.imprimirColunaNomes(epochNumber++);
			indicadorErro = false;
			erro = 0;
			for(int x = 0; x < data.length; x++) {
				double pesoSomado = perceptron.calcularSomaPesos(data[x][0], pesos);
				int resultado = perceptron.aplicarFuncaoAtivacao(pesoSomado);
				erro = data[x][1][0] - resultado;
				if (erro != 0) indicadorErro = true;
				pesosAjustados = perceptron.ajustarPesos(data[x][0], pesos, erro);
				driver.imprimirVetor(data[x], pesos, resultado, erro, pesoSomado, pesosAjustados);
				pesos = pesosAjustados;
			}
		}
	}
	
	public void imprimirColunaNomes(int epochNumber) {
		System.out.println("\n=================================================Epoch # " +epochNumber+" ===================================================");
		System.out.println("   w1   |   w2   | x1 | x2 | Resultado desejado | Resultado | Erro | Peso somado | w1 ajustado | w2 ajustado");
		System.out.println("************************************************************************************************************");
	}
	
	public void imprimirVetor(int[][] data, double[] pesos, int resultado, double erro, double pesoSomado, double[] pesosAjustados){
		
		System.out.println("  " + String.format("%.2f", pesos[0]) + " | " + String.format("%.2f", pesos[1]) + " | " + data[0][0] + " | " + data[0][1] + 
						   "  |       " + data[1][0] + "       |  " + resultado + "    | " + erro + "   |      " + String.format("%.2f ", pesoSomado) +
						   "      |     " + String.format("%.2f", pesosAjustados[0]) + "     | " + String.format("%.2f", pesosAjustados[1]));
		
	}
	
}


// começar aula 2 - https://www.youtube.com/watch?v=ZUFdrvQFlwE