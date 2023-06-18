using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Statistics.Distributions.Multivariate;
using Accord.Statistics.Distributions;
using Accord.Statistics;
using Xunit;
using MGroup.Stochastic.Tests.SupportiveClasses;
using MGroup.Stochastic.Bayesian;
using Accord.Statistics.Distributions.Univariate;
using System.IO;

namespace MGroup.Stochastic.Tests
{
	public class ConcreteHierarchicalBayesian
	{
		[Fact]
		public static void RunSimulation()
		{
			var model = new ConcreteHierarchicalModels();
			model.InitializeModels();
			var cementResponse = model.FormulateCementProblem(new double[] { 10, 1, 0.1 });
			var mortarResponse = model.FormulateMortarProblem(new double[] { 10, 1, 0.1 });
			var concreteResponse = model.FormulateConcreteProblem(new double[] { 10, 1, 0.1 });
			//var mean_hyperprior = new MultivariateUniformDistribution(new double[] { 10 }, new double[] { 30 });
			//var var_hyperprior = new MultivariateUniformDistribution(new double[] { 1 }, new double[] { 10 });
			//var lower_bound_hyperprior = new MultivariateUniformDistribution(new double[] { 0 }, new double[] { 1 });
			//var upper_bound_hyperprior = new MultivariateUniformDistribution(new double[] { 1 }, new double[] { 2 });
			//var hyperprior = new Dictionary<string, ISampleableDistribution<double[]>>();
			//hyperprior.Add("mean", mean_hyperprior);
			//hyperprior.Add("var", var_hyperprior);
			//var measurementValues = new double[3] { 9.012012689089184, 4.440024343431458, 3.1314475332399594 };
			//var measurementError = new double[3] { 0.1, 0.1, 0.1 };
			//var bayesianInstance = new HierarchicalBayesianUpdate(model.FormulateProblem, prior, hyperprior, measurementValues, measurementError);
			//var sampler = new TransitionalMarkovChainMonteCarlo(3, bayesianInstance.LikelihoodModel, bayesianInstance.PriorDistributionGenerator, scalingFactor: 0.2);
			//var samples = sampler.GenerateSamples(10000);
			//var mean = samples.Mean(0);
			//var std = samples.StandardDeviation();
			//Assert.True(Math.Abs(mean - 1.6) < 0.05);
			//Assert.True(Math.Abs(std[0] - 0.45) < 0.1);
			//var model = new ConcreteHierarchicalModels();

			var cementPrior = new MultivariateUniformDistribution(new double[] { 10 }, new double[] { 30 });
			var cementMeasurementValues = new double[1] { 9.012012689089184 };
			var cementMeasurementError = new double[1] { 0.1 };
			var cementBayesianInstance = new BayesianUpdate(model.FormulateCementProblem, cementPrior, cementMeasurementValues, cementMeasurementError);
			var cementSampler = new TransitionalMarkovChainMonteCarlo(1, cementBayesianInstance.LikelihoodModel, cementBayesianInstance.PriorDistributionGenerator, scalingFactor: 0.2);

			var mortarPrior = new MultivariateUniformDistribution(new double[] { 10 }, new double[] { 30 });
			var mortarMeasurementValues = new double[1] { 4.440024343431458 };
			var mortarMeasurementError = new double[1] { 0.1 };
			var mortarBayesianInstance = new BayesianUpdate(model.FormulateMortarProblem, mortarPrior, mortarMeasurementValues, mortarMeasurementError);
			var mortarSampler = new TransitionalMarkovChainMonteCarlo(1, mortarBayesianInstance.LikelihoodModel, mortarBayesianInstance.PriorDistributionGenerator, scalingFactor: 0.2);

			var concretePrior = new MultivariateUniformDistribution(new double[] { 10 }, new double[] { 30 });
			var concreteMeasurementValues = new double[1] { 3.1314475332399594 };
			var concreteMeasurementError = new double[1] { 0.1 };
			var concreteBayesianInstance = new BayesianUpdate(model.FormulateConcreteProblem, concretePrior, concreteMeasurementValues, concreteMeasurementError);
			var concreteSampler = new TransitionalMarkovChainMonteCarlo(1, concreteBayesianInstance.LikelihoodModel, concreteBayesianInstance.PriorDistributionGenerator, scalingFactor: 0.2);

			var numParameterSamples = 200;
			//var cementSamples = cementSampler.GenerateSamples(numParameterSamples);
			//var mortarSamples = mortarSampler.GenerateSamples(numParameterSamples);
			//var concreteSamples = concreteSampler.GenerateSamples(numParameterSamples);
			double[,] cementSamples = new double[200, 1];
			double[,] mortarSamples = new double[200, 1];
			double[,] concreteSamples = new double[200, 1];

			var cement = File.ReadAllText("cement.txt");

			int ii = 0;		
			foreach (var row in cement.Split('\n'))
			{
				cementSamples[ii, 0] = double.Parse(row);
				ii++;
			}

			var mortar = File.ReadAllText("mortar.txt");

			ii = 0;
			foreach (var row in mortar.Split('\n'))
			{
				mortarSamples[ii, 0] = double.Parse(row);
				ii++;
			}

			var concrete = File.ReadAllText("concrete.txt");

			ii = 0;
			foreach (var row in concrete.Split('\n'))
			{
				concreteSamples[ii, 0] = double.Parse(row);
				ii++;
			}

			var hyperparameterPrior = new MultivariateUniformDistribution(new double[] { 5, 21 }, new double[] { 19, 35 });
			var hyperparameterSampler = new TransitionalMarkovChainMonteCarlo(2, HyperparameterLikelihoodModel, hyperparameterPrior.Generate, scalingFactor: 0.2);

			var numHyperparameterSamples = 50000;
			var hyperparameterSamples = hyperparameterSampler.GenerateSamples(numHyperparameterSamples);

			var mixtureComponents = new MultivariateUniformDistribution[numHyperparameterSamples];
			for (int i = 0; i < numHyperparameterSamples; i++)
			{
				mixtureComponents[i] = new MultivariateUniformDistribution(new double[] { hyperparameterSamples[i, 0] }, new double[] { hyperparameterSamples[i, 1] });
			}
			var mixture = new MultivariateMixture<MultivariateUniformDistribution>(mixtureComponents);

			var numNewParameterSamples = 50000;
			var newParameterSamples = To2D(mixture.Generate(numNewParameterSamples));


			double HyperparameterLikelihoodModel(double[] sample)
			{
				var cementLikelihoodValue = 0d;
				var tempDistribution = new MultivariateUniformDistribution(new double[] { sample[0] }, new double[] { sample[1] });
				for (int i = 0; i < cementSamples.GetLength(0); i++)
				{
					cementLikelihoodValue += tempDistribution.ProbabilityDensityFunction(cementSamples[i, 0]);
				}
				cementLikelihoodValue = cementLikelihoodValue / cementSamples.GetLength(0);
				var mortarLikelihoodValue = 0d;
				for (int i = 0; i < mortarSamples.GetLength(0); i++)
				{
					mortarLikelihoodValue += tempDistribution.ProbabilityDensityFunction(mortarSamples[i, 0]);
				}
				mortarLikelihoodValue = mortarLikelihoodValue / mortarSamples.GetLength(0);
				var concreteLikelihoodValue = 0d;
				for (int i = 0; i < concreteSamples.GetLength(0); i++)
				{
					concreteLikelihoodValue += tempDistribution.ProbabilityDensityFunction(concreteSamples[i, 0]);
				}
				concreteLikelihoodValue = concreteLikelihoodValue / concreteSamples.GetLength(0);

				return cementLikelihoodValue*mortarLikelihoodValue*concreteLikelihoodValue;
			}
		}

		private static double[,] To2D(double[][] source)
		{
			int FirstDim = source.Length;
			int SecondDim = source[0].Length;

			var result = new double[FirstDim, SecondDim];
			for (int i = 0; i < FirstDim; i++)
			{
				var tempSecondDim = source[i].Length;
				for (int j = 0; j < tempSecondDim; j++)
					result[i, j] = source[i][j];
				if (tempSecondDim < SecondDim)
					for (int j = tempSecondDim; j < SecondDim; j++)
					{
						result[i, j] = 0;
					}
			}

			return result;
		}

	}
}
