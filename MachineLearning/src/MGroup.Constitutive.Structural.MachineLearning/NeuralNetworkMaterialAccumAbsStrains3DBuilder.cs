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
using MGroup.Constitutive.Structural.Cohesive;
using DotNumerics.Optimization.TN;
using MGroup.Stochastic.Structural;

//using MatlabWriter = MathNet.Numerics.Data.Matlab.MatlabWriter;

//[assembly: SuppressXUnitOutputException]

namespace MGroup.Constitutive.Structural.MachineLearning
{
	public class NeuralNetworkMaterialAccumAbsStrains3DBuilder
	{
		string SpecPath;
		string InputFileName;
		string OutputFileName;

		public NeuralNetworkMaterialAccumAbsStrains3DBuilder()
		{
			//path definitions
			SpecPath = @"MsolveOutputs\NeuralNetworks\StrainStressData";
			InputFileName = "StrainData_acc_strains.txt";
			OutputFileName = "StressData_acc_strains.txt";
		}

		public void GenerateStrainStressData()
		{
			//model properties/
			var rnd = new Random();
			var matrixMaterial = new VonMisesMaterial3D(youngModulus: 20, poissonRatio: 0.2, yieldStress: 0.1, hardeningRatio: 2);
			//var matrixMaterial = new ElasticMaterial3D(youngModulus: 20, poissonRatio: 0.2);
			//var matrixMaterial = new DruckerPrager3DFunctional(youngModulus: 3.5, poissonRatio: 0.4, friction: 20, dilation: 20, cohesion: 0.01, hardeningFunc: x => (0.01 + 0.01 * (1 - Math.Exp(-500 * x)))); //(0.02 * (1 - Math.Exp(-100 * x)))
			//var cohesiveMaterial = new BondSlipMaterial(K_el, K_pl, 100.0, T_max, new double[2], new double[2], 1e-3);
			var numOfInclusions = 260;
			var rveBuilder = new CntReinforcedElasticNanocomposite(numOfInclusions, matrixMaterial);
			rveBuilder.readFromText = true;
			var microstructure = new Microstructure3D<SymmetricCscMatrix>(rveBuilder, false, 1, new SuiteSparseSolverPrefernce());
			//var microstructure = new Microstructure3D<SkylineMatrix>(rveBuilder, false, 1, new SkylineSolverPrefernce());
			var maxParameterValues = new double[2] { 20, 0.2 };
			var minParameterValues = new double[2] { 20, 0.2 };

			//analysis properties
			var numOfSolutions = 100;
			var incrementsPerSolution = 40;
			var maxLimitStrain = new double[6] { 0.05, 0.05, 0.05, 0.05, 0.05, 0.05 };
			var minLimitStrain = new double[6] { -0.05, -0.05, -0.05, -0.05, -0.05, -0.05 };
			var maxPerturbationStrain = new double[6] { 0.001, 0.001, 0.001, 0.001, 0.001, 0.001 };
			var minPerturbationStrain = new double[6] { -0.001, -0.001, -0.001, -0.001, -0.001, -0.001 };

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
			var checkNaN = false;
			//var GPsampler = new DiscreteGaussianProcess(6, incrementsPerSolution + 1, 0.5*maxPerturbationStrain[0], 1, true, x => x / incrementsPerSolution * 0.01);
			for (int num_solution = 0; num_solution < numOfSolutions; num_solution++)
			{
				checkNaN = false;
				var TotalMacroStrain = new double[maxLimitStrain.Length];
				var gp_sample = new double[maxLimitStrain.Length][];
				for (int i = 0; i < maxLimitStrain.Length; i++)
				{
					TotalMacroStrain[i] = rnd.NextDouble() * (maxLimitStrain[i] - minLimitStrain[i]) + minLimitStrain[i];
					var gpSampler = new DiscreteGaussianProcess(incrementsPerSolution + 1, 0.4 * maxPerturbationStrain[i], 1.5, true, x => x / incrementsPerSolution * TotalMacroStrain[i]);
					gp_sample[i] = gpSampler.Generate();
				}
				var parameterValues = new double[maxParameterValues.Length];
				for (int i = 0; i < maxParameterValues.Length; i++)
				{
					parameterValues[i] = rnd.NextDouble() * (maxParameterValues[i] - minParameterValues[i]) + minParameterValues[i];
				}
				matrixMaterial = new VonMisesMaterial3D(youngModulus: 20, poissonRatio: 0.2, yieldStress: 0.1, hardeningRatio: 2);
				//matrixMaterial = new ElasticMaterial3D(youngModulus: 20, poissonRatio: 0.2);
				//matrixMaterial = new DruckerPrager3DFunctional(youngModulus: 3.5, poissonRatio: 0.4, friction: 20, dilation: 20, cohesion: 0.01, hardeningFunc: x => (0.01 + 0.01 * (1 - Math.Exp(-500 * x))));
				rveBuilder =
					new CntReinforcedElasticNanocomposite(numOfInclusions, matrixMaterial); //{ K_el = 20, K_pl = 2, T_max = 0.2, };
				rveBuilder.readFromText = true;
				microstructure = new Microstructure3D<SymmetricCscMatrix>(rveBuilder, false, 1, new SuiteSparseSolverPrefernce());
				//microstructure = new Microstructure3D<SkylineMatrix>(rveBuilder, false, 1, new SkylineSolverPrefernce());

				//TotalMacroStrain = new double[6] { 0.02, 0.00, 0.00, 0.00, 0.00, 0.00 };
				var perturbation = new double[maxPerturbationStrain.Length];
				var IncrMacroStrain = new double[6];
				var MacroStrain = new double[maxLimitStrain.Length];
				var prevMacroStrain = new double[maxLimitStrain.Length];
				var AccAbsStrain = new double[6];
				for (int i = 0; i < incrementsPerSolution; i++)
				{
					prevMacroStrain = MacroStrain.Copy();
					for (int k = 0; k < maxPerturbationStrain.Length; k++)
					{
						perturbation[k] = rnd.NextDouble() * (maxPerturbationStrain[k] - minPerturbationStrain[k]) + minPerturbationStrain[k];
					}
					for (int ii = 0; ii < 6; ii++) { IncrMacroStrain[ii] = (i + 1) * TotalMacroStrain[ii] / incrementsPerSolution; }
					for (int k = 0; k < maxPerturbationStrain.Length; k++)
					{
						//MacroStrain[k] = IncrMacroStrain[k] + perturbation[k];
						MacroStrain[k] = gp_sample[k][i + 1];
					}
					microstructure.UpdateConstitutiveMatrixAndEvaluateResponse(new double[6] { MacroStrain[0], MacroStrain[1], MacroStrain[2], MacroStrain[3], MacroStrain[4], MacroStrain[5] });

					double[] MacroStress = new double[6] { microstructure.Stresses[0], microstructure.Stresses[1], microstructure.Stresses[2], microstructure.Stresses[3], microstructure.Stresses[4], microstructure.Stresses[5] };

					for(int ii=0; ii< 6; ii++)
					{
						if (double.IsNaN(MacroStress[ii])==true)
						{
							checkNaN = true;
							break;
						}
					}
					if (checkNaN == true)
					{ break; }
					microstructure.CreateState();

					for (int ii = 0; ii < 6; ii++) { AccAbsStrain[ii] = AccAbsStrain[ii] + Math.Abs(MacroStrain[ii] - prevMacroStrain[ii]); }

					Input[num_solution * incrementsPerSolution + i] = new double[12] {
																						MacroStrain[0], MacroStrain[1], MacroStrain[2], MacroStrain[3], MacroStrain[4], MacroStrain[5],
																						AccAbsStrain[0], AccAbsStrain[1], AccAbsStrain[2], AccAbsStrain[3], AccAbsStrain[4], AccAbsStrain[5] };
					Output[num_solution * incrementsPerSolution + i] = new double[6] { MacroStress[0], MacroStress[1], MacroStress[2], MacroStress[3], MacroStress[4], MacroStress[5] }; //homogeneousRveBuilder1.K_el, homogeneousRveBuilder1.K_pl, homogeneousRveBuilder1.T_max,

					using (var writer = new StreamWriter(inputFile, append)) // append mode to continue from previous increment
					{
						writer.WriteLine($"{Input[num_solution * incrementsPerSolution + i][0]}, {Input[num_solution * incrementsPerSolution + i][1]}, {Input[num_solution * incrementsPerSolution + i][2]}, " +
							$"{Input[num_solution * incrementsPerSolution + i][3]}, {Input[num_solution * incrementsPerSolution + i][4]}, {Input[num_solution * incrementsPerSolution + i][5]}, " +
							$"{Input[num_solution * incrementsPerSolution + i][6]}, {Input[num_solution * incrementsPerSolution + i][7]}, {Input[num_solution * incrementsPerSolution + i][8]}, " +
							$"{Input[num_solution * incrementsPerSolution + i][9]}, {Input[num_solution * incrementsPerSolution + i][10]}, {Input[num_solution * incrementsPerSolution + i][11]}");//, {Input[num_solution * incrementsPerSolution + i][6]}, " +																																															   //$"{Input[num_solution * incrementsPerSolution + i][7]}, {Input[num_solution * incrementsPerSolution + i][8]}");
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
					new InputLayer(new int[]{12}),
					new DenseLayer(30, ActivationType.TanH),
					new DenseLayer(30, ActivationType.TanH),
					new DenseLayer(6, ActivationType.Linear)
		},
		3000, batchSize: 128);

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
			var netPathName = "network_architecture_acc_strains";
			var weightsPathName = "trained_weights_acc_strains";
			var normalizationPathName = "normalization_acc_strains";

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
