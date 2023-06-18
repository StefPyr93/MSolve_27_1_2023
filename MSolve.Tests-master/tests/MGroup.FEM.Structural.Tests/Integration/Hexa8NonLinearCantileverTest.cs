using System.Collections.Generic;
using MGroup.MSolve.Discretization.Entities;
using MGroup.MSolve.Discretization.Dofs;
using MGroup.Constitutive.Structural;
using MGroup.NumericalAnalyzers;
using MGroup.NumericalAnalyzers.Logging;
using MGroup.Solvers.Direct;
using MGroup.NumericalAnalyzers.Discretization.NonLinear;
using MGroup.FEM.Structural.Tests.ExampleModels;
using MGroup.FEM.Structural.Tests.Commons;
using Xunit;

namespace MGroup.FEM.Structural.Tests.Integration
{

	public static class Hexa8NonLinearCantileverTest
	{
		[Fact]
		private static void RunTest()
		{
			var model = Hexa8NonLinearCantileverExample.CreateModel();
			var computedDisplacements = SolveModel(model);
			Assert.True(Utilities.AreDisplacementsSame(Hexa8NonLinearCantileverExample.GetExpectedDisplacements(), computedDisplacements, tolerance: 1E-13));
		}

		private static TotalDisplacementsPerIterationLog SolveModel(Model model)
		{
			var solverFactory = new SkylineSolver.Factory();
			//var solverFactory = new DenseMatrixSolver.Factory() { IsMatrixPositiveDefinite = false };
			var algebraicModel = solverFactory.BuildAlgebraicModel(model);
			var solver = solverFactory.BuildSolver(algebraicModel);
			var problem = new ProblemStructural(model, algebraicModel);

			var loadControlAnalyzerBuilder = new LoadControlAnalyzer.Builder(algebraicModel, solver, problem, numIncrements: 10)
			{
				ResidualTolerance = 1E-2,
				MaxIterationsPerIncrement = 100,
				NumIterationsForMatrixRebuild = 1
			};
			var loadControlAnalyzer = loadControlAnalyzerBuilder.Build();
			var staticAnalyzer = new StaticAnalyzer(algebraicModel, problem, loadControlAnalyzer);
			
			loadControlAnalyzer.TotalDisplacementsPerIterationLog = new TotalDisplacementsPerIterationLog(
				new List<(INode node, IDofType dof)>()
				{
					(model.NodesDictionary[5], StructuralDof.TranslationX),
					(model.NodesDictionary[8], StructuralDof.TranslationX),
					(model.NodesDictionary[12], StructuralDof.TranslationX),
					(model.NodesDictionary[16], StructuralDof.TranslationX),
					(model.NodesDictionary[20], StructuralDof.TranslationX)
				}, algebraicModel
			);

			staticAnalyzer.Initialize();
			staticAnalyzer.Solve();

			return loadControlAnalyzer.TotalDisplacementsPerIterationLog;
		}
	}
}
