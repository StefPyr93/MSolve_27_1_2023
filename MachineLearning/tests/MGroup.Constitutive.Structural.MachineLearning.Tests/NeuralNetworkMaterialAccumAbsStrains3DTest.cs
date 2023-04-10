using System;
using MGroup.MachineLearning.TensorFlow.NeuralNetworks;
using MGroup.Constitutive.Structural.Continuum;
using MGroup.LinearAlgebra.Matrices;
using Xunit;
using System.Reflection;
using MGroup.MachineLearning;
using MGroup.Stochastic.Structural;
using MGroup.MSolve.MultiscaleAnalysis;
using MGroup.Multiscale;
using MiMsolve.SolutionStrategies;

namespace MGroup.Constitutive.Structural.MachineLearning.Tests
{
	public static class NeuralNetworkMaterialAccumAbsStrains3DTest
	{
		[Fact]
		public static void RunTest()
		{
			// these files are used to generate an already trained FeedForwardNeuralNetwork which was created using strain-stress pairs from an ElasticMaterial3D(youngModulus:20, poissonRatio:0.2)
			string initialPath = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location).Split(new string[] { "\\bin" }, StringSplitOptions.None)[0];
			var folderName = "SavedFiles";
			var netPathName = "network_architecture_acc_strains";
			netPathName = Path.Combine(initialPath, folderName, netPathName);
			var weightsPathName = "trained_weights_acc_strains";
			weightsPathName = Path.Combine(initialPath, folderName, weightsPathName);
			var normalizationPathName = "normalization_acc_strains";
			normalizationPathName = Path.Combine(initialPath, folderName, normalizationPathName);

			var neuralNetwork = new FeedForwardNeuralNetwork();
			neuralNetwork.LoadNetwork(netPathName, weightsPathName, normalizationPathName);

			//var neuralNetworkMaterial = new NeuralNetworkMaterialAccumAbsStrains3D(neuralNetwork, new double[0]);
			//var plasticMaterial = new VonMisesMaterial3D(20, 0.2, 0.1, 2);

			CheckNeuralNetworkMaterialAccuracy(neuralNetwork);
		}

		private static void CheckNeuralNetworkMaterialAccuracy(FeedForwardNeuralNetwork neuralNetwork)
		{
			var numOfSolutions = 10;
			var incrementsPerSolution = 40;
			var maxLimitStrain = new double[6] { 0.05, 0.05, 0.05, 0.05, 0.05, 0.05 };
			var minLimitStrain = new double[6] { -0.05, -0.05, -0.05, -0.05, -0.05, -0.05 };
			var maxPerturbationStrain = new double[6] { 0.001, 0.001, 0.001, 0.001, 0.001, 0.001 };
			var minPerturbationStrain = new double[6] { -0.001, -0.001, -0.001, -0.001, -0.001, -0.001 };
			var rnd = new Random();
			var strains = new double[numOfSolutions * incrementsPerSolution, 6];
			var stressesNeuralNetwork = new double[numOfSolutions * incrementsPerSolution, 6];
			var constitutiveNeuralNetwork = new Matrix[numOfSolutions * incrementsPerSolution];
			var stressesPlastic= new double[numOfSolutions * incrementsPerSolution, 6];
			var constitutivePlastic = new Matrix[numOfSolutions * incrementsPerSolution];
			var TotalMacroStrain = new double[maxLimitStrain.Length];
			for (int k = 0; k < numOfSolutions; k++)
			{
				var gp_sample = new double[maxLimitStrain.Length][];
				for (int i = 0; i < maxLimitStrain.Length; i++)
				{
					TotalMacroStrain[i] = rnd.NextDouble() * (maxLimitStrain[i] - minLimitStrain[i]) + minLimitStrain[i];
					var gpSampler = new DiscreteGaussianProcess(incrementsPerSolution + 1, 0.4 * maxPerturbationStrain[i], 1.5, true, x => x / incrementsPerSolution * TotalMacroStrain[i]);
					gp_sample[i] = gpSampler.Generate();
				}
				var plasticMaterial = new DruckerPrager3DFunctional(youngModulus: 3.5, poissonRatio: 0.4, friction: 20, dilation: 20, cohesion: 0.01, hardeningFunc: x => (0.01 + 0.01 * (1 - Math.Exp(-500 * x))));
				var rveBuilder2 =
					new CntReinforcedElasticNanocomposite(260, plasticMaterial); //{ K_el = 20, K_pl = 2, T_max = 0.2, };
				rveBuilder2.readFromText = true;
				var microstructure2 = new Microstructure3D<SymmetricCscMatrix>(rveBuilder2, false, 1, new SuiteSparseSolverPrefernce());
				var perturbation2 = new double[maxPerturbationStrain.Length];
				var IncrMacroStrain2 = new double[6];
				var MacroStrain2 = new double[maxLimitStrain.Length];
				var prevMacroStrain2 = new double[maxLimitStrain.Length];
				var IncrStrain2 = new double[6];
				for (int i = 0; i < incrementsPerSolution; i++)
				{
					prevMacroStrain2 = MacroStrain2.Copy();
					for (int ii = 0; ii < MacroStrain2.Length; ii++)
					{
						//MacroStrain[k] = IncrMacroStrain[k] + perturbation[k];
						MacroStrain2[ii] = gp_sample[ii][i + 1];
					}
					microstructure2.UpdateConstitutiveMatrixAndEvaluateResponse(new double[6] { MacroStrain2[0], MacroStrain2[1], MacroStrain2[2], MacroStrain2[3], MacroStrain2[4], MacroStrain2[5] });

					double[] MacroStress2 = new double[6] { microstructure2.Stresses[0], microstructure2.Stresses[1], microstructure2.Stresses[2], microstructure2.Stresses[3], microstructure2.Stresses[4], microstructure2.Stresses[5] };

					for (int j = 0; j < 6; j++)
					{
						stressesPlastic[k * incrementsPerSolution + i, j] = microstructure2.Stresses[j];
					}
					constitutivePlastic[k * incrementsPerSolution + i] = (Matrix)microstructure2.ConstitutiveMatrix.Copy();
					microstructure2.CreateState();
				}

				var neuralNetworkMaterial = new NeuralNetworkMaterialAccumAbsStrains3D(neuralNetwork, new double[0]);
				//var rveBuilder =
				//	new CntReinforcedElasticNanocomposite(0, neuralNetworkMaterial); //{ K_el = 20, K_pl = 2, T_max = 0.2, };
				//rveBuilder.readFromText = false;
				//var microstructure = new Microstructure3D<SkylineMatrix>(rveBuilder, false, 1, new SkylineSolverPrefernce());
				//var perturbation = new double[maxPerturbationStrain.Length];
				var IncrMacroStrain = new double[6];
				var MacroStrain = new double[maxLimitStrain.Length];
				var prevMacroStrain = new double[maxLimitStrain.Length];
				//var IncrStrain = new double[6];
				for (int i = 0; i < incrementsPerSolution; i++)
				{
					prevMacroStrain = MacroStrain.Copy();
					for (int ii = 0; ii < MacroStrain.Length; ii++)
					{
						//MacroStrain[k] = IncrMacroStrain[k] + perturbation[k];
						MacroStrain[ii] = gp_sample[ii][i + 1];
					}
					for (int j = 0; j < 6; j++)
					{
						strains[k * incrementsPerSolution + i, j] = MacroStrain[j];
					}
					for (int ii = 0; ii < 6; ii++) { IncrMacroStrain[ii] = MacroStrain[ii] - prevMacroStrain[ii]; }
					neuralNetworkMaterial.UpdateConstitutiveMatrixAndEvaluateResponse(new double[6] { IncrMacroStrain[0], IncrMacroStrain[1], IncrMacroStrain[2], IncrMacroStrain[3], IncrMacroStrain[4], IncrMacroStrain[5] });

					double[] MacroStress = new double[6] { neuralNetworkMaterial.Stresses[0], neuralNetworkMaterial.Stresses[1], neuralNetworkMaterial.Stresses[2], neuralNetworkMaterial.Stresses[3], neuralNetworkMaterial.Stresses[4], neuralNetworkMaterial.Stresses[5] };

					for (int j = 0; j < 6; j++)
					{
						stressesNeuralNetwork[k * incrementsPerSolution + i, j] = neuralNetworkMaterial.Stresses[j];
					}
					constitutiveNeuralNetwork[k * incrementsPerSolution + i] = (Matrix)neuralNetworkMaterial.ConstitutiveMatrix.Copy();
					neuralNetworkMaterial.CreateState();
				}

			}

			var stressDeviation = 0d;
			var constitutiveDeviation = 0d;
			for (int k = 0; k < numOfSolutions; k++)
			{
				for (int i = 0; i < incrementsPerSolution; i++)
				{
					for (int j1 = 0; j1 < 6; j1++)
					{
						stressDeviation += Math.Pow((stressesNeuralNetwork[k * incrementsPerSolution + i, j1] - stressesPlastic[k * incrementsPerSolution + i, j1]), 2);
						for (int j2 = 0; j2 < 6; j2++)
						{
							constitutiveDeviation += Math.Pow((constitutiveNeuralNetwork[k * incrementsPerSolution + i][j1, j2] - constitutivePlastic[k * incrementsPerSolution + i][j1, j2]), 2);
						}
					}
				}
			}

			stressDeviation = stressDeviation / (incrementsPerSolution * numOfSolutions);
			constitutiveDeviation = constitutiveDeviation / (incrementsPerSolution * numOfSolutions);

			Assert.True(stressDeviation < 1e-6 && constitutiveDeviation < 2e-1);
		}
	}
}