using System;
using System.IO;
using System.Linq;
using MGroup.Constitutive.Structural.Continuum;
using MGroup.LinearAlgebra.Matrices;
using MGroup.MachineLearning.Preprocessing;
using MGroup.MachineLearning.TensorFlow.NeuralNetworks;
using MGroup.MSolve.MultiscaleAnalysis;
using MGroup.Multiscale;
using static Tensorflow.KerasApi;

using MGroup.MachineLearning;
using MiMsolve.SolutionStrategies;
using MGroup.MachineLearning.TensorFlow.KerasLayers;
using Tensorflow.Operations.Activation;

//using MatlabWriter = MathNet.Numerics.Data.Matlab.MatlabWriter;

//[assembly: SuppressXUnitOutputException]

namespace MGroup.Constitutive.Structural.MachineLearning
{
	public class NeuralNetworkMaterialBuilder
	{
		string SpecPath;
		string InputFileName;
		string OutputFileName;

		public NeuralNetworkMaterialBuilder()
		{
			//path definitions
			SpecPath = @"MsolveOutputs\NeuralNetworks\StrainStressData";
			InputFileName = "StrainData.txt";
			OutputFileName = "StressData.txt";
		}

		public void GenerateStrainStressData()
		{
			//model properties
			var rnd = new Random();
			var matrixMaterial = new ElasticMaterial3D(youngModulus: 20, poissonRatio: 0.2);
			var numOfInclusions = 0;
			var rveBuilder = new CntReinforcedElasticNanocomposite(numOfInclusions, matrixMaterial);
			var microstructure = new Microstructure3D<SkylineMatrix>(rveBuilder, false, 1, new SkylineSolverPrefernce());
			var maxParameterValues = new double[2] { 20, 0.2 };
			var minParameterValues = new double[2] { 20, 0.2 };

			//analysis properties
			var numOfSolutions = 500;
			var incrementsPerSolution = 5;
			var maxLimitStrain = new double[6] { 0.02, 0.02, 0.02, 0.02, 0.02, 0.02 };
			var minLimitStrain = new double[6] { 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 };

			//run analyses and save input-output pairs
			var BasePath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
			var pathName = Path.Combine(BasePath, SpecPath);

			string InputExtension = Path.GetExtension(InputFileName);
			string InputfileNameOnly = Path.Combine(pathName, Path.GetFileNameWithoutExtension(InputFileName));
			string inputFile = string.Format("{0}{1}", InputfileNameOnly, InputExtension);

			string OutputExtension = Path.GetExtension(OutputFileName);
			string OutputfileNameOnly = Path.Combine(pathName, Path.GetFileNameWithoutExtension(OutputFileName));
			string outputFile = string.Format("{0}{1}", OutputfileNameOnly, OutputExtension);

			bool append = false;

			double[][] Input = new double[numOfSolutions * incrementsPerSolution][];
			double[][] Output = new double[numOfSolutions * incrementsPerSolution][];

			//Microstructure3D<SkylineMatrix> microstructure = new Microstructure3D<SkylineMatrix>(homogeneousRveBuilder1, false, 1, new SkylineSolverPrefernce());

			for (int num_solution = 0; num_solution < numOfSolutions; num_solution++)
			{
				var parameterValues = new double[maxParameterValues.Length];
				for (int i = 0; i < maxParameterValues.Length; i++)
				{
					parameterValues[i] = rnd.NextDouble() * (maxParameterValues[i] - minParameterValues[i]) + minParameterValues[i];
				}
				matrixMaterial = new ElasticMaterial3D(parameterValues[0], parameterValues[1]);
				var homogeneousRveBuilder1 =
					new CntReinforcedElasticNanocomposite(numOfInclusions, matrixMaterial); //{ K_el = 20, K_pl = 2, T_max = 0.2, };

				var MacroStrain = new double[maxLimitStrain.Length];
				for (int i = 0; i < maxLimitStrain.Length; i++)
				{
					MacroStrain[i] = rnd.NextDouble() * (maxLimitStrain[i] - minLimitStrain[i]) + minLimitStrain[i];
				}
				for (int i = 0; i < incrementsPerSolution; i++)
				{
					var IncrMacroStrain = new double[6];
					for (int ii = 0; ii < 6; ii++) { IncrMacroStrain[ii] = (i + 1) * MacroStrain[ii] / incrementsPerSolution; }
					microstructure.UpdateConstitutiveMatrixAndEvaluateResponse(new double[6] { IncrMacroStrain[0], IncrMacroStrain[1], IncrMacroStrain[2], IncrMacroStrain[3], IncrMacroStrain[4], IncrMacroStrain[5] });

					double[] IncrMacroStress = new double[6] { microstructure.Stresses[0], microstructure.Stresses[1], microstructure.Stresses[2], microstructure.Stresses[3], microstructure.Stresses[4], microstructure.Stresses[5] };

					microstructure.CreateState();

					Input[num_solution * incrementsPerSolution + i] = new double[6] {
																						IncrMacroStrain[0], IncrMacroStrain[1], IncrMacroStrain[2], IncrMacroStrain[3], IncrMacroStrain[4], IncrMacroStrain[5] };
					Output[num_solution * incrementsPerSolution + i] = new double[6] { IncrMacroStress[0], IncrMacroStress[1], IncrMacroStress[2], IncrMacroStress[3], IncrMacroStress[4], IncrMacroStress[5] }; //homogeneousRveBuilder1.K_el, homogeneousRveBuilder1.K_pl, homogeneousRveBuilder1.T_max,

					using (var writer = new StreamWriter(inputFile, append)) // append mode to continue from previous increment
					{
						writer.WriteLine($"{Input[num_solution * incrementsPerSolution + i][0]}, {Input[num_solution * incrementsPerSolution + i][1]}, {Input[num_solution * incrementsPerSolution + i][2]}, " +
							$"{Input[num_solution * incrementsPerSolution + i][3]}, {Input[num_solution * incrementsPerSolution + i][4]}, {Input[num_solution * incrementsPerSolution + i][5]}");//, {Input[num_solution * incrementsPerSolution + i][6]}, " +																																															   //$"{Input[num_solution * incrementsPerSolution + i][7]}, {Input[num_solution * incrementsPerSolution + i][8]}");
					}

					using (var writer = new StreamWriter(outputFile, append)) // append mode to continue from previous increment
					{
						writer.WriteLine($"{Output[num_solution * incrementsPerSolution + i][0]}, {Output[num_solution * incrementsPerSolution + i][1]}, {Output[num_solution * incrementsPerSolution + i][2]}, " +
							$"{Output[num_solution * incrementsPerSolution + i][3]}, {Output[num_solution * incrementsPerSolution + i][4]}, {Output[num_solution * incrementsPerSolution + i][5]}");
					}
					append = true;
				}
			}
		}

		public void PerformRveSolutions()
		{
			//LinearAlgebra.LibrarySettings.LinearAlgebraProviders = LinearAlgebra.LinearAlgebraProviderChoice.MKL;

			

		}

		public INeuralNetwork TrainNeuralNetwork()
		{
			var neuralNetwork = new FeedForwardNeuralNetwork(new MinMaxNormalization(), new MinMaxNormalization(),
		new MGroup.MachineLearning.TensorFlow.Keras.Optimizers.Adam(dataType: Tensorflow.TF_DataType.TF_DOUBLE, learning_rate: 0.001f),
		keras.losses.MeanSquaredError(), new INetworkLayer[]
		{
					new InputLayer(new int[]{6}),
					new DenseLayer(30, ActivationType.TanH),
					new DenseLayer(30, ActivationType.TanH),
					new DenseLayer(6, ActivationType.Linear)
		},
		2000, batchSize: 128);

			var BasePath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
			var pathName = Path.Combine(BasePath, SpecPath);

			string InputExtension = Path.GetExtension(InputFileName);
			string InputfileNameOnly = Path.Combine(pathName, Path.GetFileNameWithoutExtension(InputFileName));
			string inputFile = string.Format("{0}{1}", InputfileNameOnly, InputExtension);

			string OutputExtension = Path.GetExtension(OutputFileName);
			string OutputfileNameOnly = Path.Combine(pathName, Path.GetFileNameWithoutExtension(OutputFileName));
			string outputFile = string.Format("{0}{1}", OutputfileNameOnly, OutputExtension);

			//var strainData = File.ReadAllText(inputFile);
			//var stressData = File.ReadAllText(outputFile);

			string[] strainDataLines = File.ReadAllLines(inputFile);
			double[,] strainData = new double[strainDataLines.Length, strainDataLines[0].Split(',').Length];
			for (int i = 0; i < strainDataLines.Length; ++i)
			{
				string line = strainDataLines[i];
				for (int j = 0; j < strainData.GetLength(1); ++j)
				{
					string[] split = line.Split(',');
					strainData[i, j] = Convert.ToDouble(split[j]);
				}
			}

			string[] stressDataLines = File.ReadAllLines(outputFile);
			double[,] stressData = new double[stressDataLines.Length, stressDataLines[0].Split(',').Length];
			for (int i = 0; i < stressDataLines.Length; ++i)
			{
				string line = stressDataLines[i];
				for (int j = 0; j < stressData.GetLength(1); ++j)
				{
					string[] split = line.Split(',');
					stressData[i, j] = Convert.ToDouble(split[j]);
				}
			}

			SpecPath = @"MsolveOutputs\NeuralNetworks";
			var netPathName = "network_architecture";
			var weightsPathName = "trained_weights";
			var normalizationPathName = "normalization";

			pathName = Path.Combine(BasePath, SpecPath);

			InputExtension = Path.GetExtension(netPathName);
			InputfileNameOnly = Path.Combine(pathName, Path.GetFileNameWithoutExtension(netPathName));
			var netPathFile = string.Format("{0}{1}", InputfileNameOnly, InputExtension);

			InputExtension = Path.GetExtension(weightsPathName);
			InputfileNameOnly = Path.Combine(pathName, Path.GetFileNameWithoutExtension(weightsPathName));
			var weightsPathFile = string.Format("{0}{1}", InputfileNameOnly, InputExtension);

			InputExtension = Path.GetExtension(normalizationPathName);
			InputfileNameOnly = Path.Combine(pathName, Path.GetFileNameWithoutExtension(normalizationPathName));
			var normalizationPathFile = string.Format("{0}{1}", InputfileNameOnly, InputExtension);

			neuralNetwork.Train(strainData, stressData, testX: null, testY: null);

			neuralNetwork.SaveNetwork(netPathFile, weightsPathFile, normalizationPathFile);

			return neuralNetwork;
		}
	}
}
