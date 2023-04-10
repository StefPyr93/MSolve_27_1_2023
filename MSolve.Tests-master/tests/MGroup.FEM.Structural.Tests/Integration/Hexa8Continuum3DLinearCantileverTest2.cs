using System.Collections.Generic;
using MGroup.Constitutive.Structural;
using MGroup.FEM.Structural.Tests.Commons;
using MGroup.FEM.Structural.Tests.ExampleModels;
using MGroup.MSolve.Discretization.Dofs;
using MGroup.MSolve.Discretization.Entities;
using MGroup.NumericalAnalyzers;
using MGroup.NumericalAnalyzers.Discretization.NonLinear;
using MGroup.NumericalAnalyzers.Logging;
using MGroup.Solvers.Direct;
using Xunit;

namespace MGroup.FEM.Structural.Tests.Integration
{
	public static class Hexa8Continuum3DLinearCantileverTest
	{
		private static List<(INode node, IDofType dof)> watchDofs = new List<(INode node, IDofType dof)>();

		[Fact]
		private static void RunTest()
		{
			var model = Hexa8Continuum3DLinearCantileverExample.CreateModel();
			var log = SolveModel(model);
			Assert.Equal(Hexa8Continuum3DLinearCantileverExample.expected_solution_node20_TranslationX, log.DOFValues[watchDofs[0].node, watchDofs[0].dof], precision: 1);
			//var computedDisplacements = SolveModel(model);
			//Assert.True(Utilities.AreDisplacementsSame(Hexa8Continuum3DLinearCantileverExample.GetExpectedDisplacements(), computedDisplacements, tolerance: 4e-1));
		}

		//private static IncrementalDisplacementsLog SolveModel(Model model)
		//{
		//	var solverFactory = new SkylineSolver.Factory();
		//	var algebraicModel = solverFactory.BuildAlgebraicModel(model);
		//	var solver = solverFactory.BuildSolver(algebraicModel);
		//	var problem = new ProblemStructural(model, algebraicModel);

		//	var loadControlAnalyzerBuilder = new LoadControlAnalyzer.Builder(algebraicModel, solver, problem, numIncrements: 100)
		//	{
		//		ResidualTolerance = 1E-5,
		//		MaxIterationsPerIncrement = 100,
		//		NumIterationsForMatrixRebuild = 1
		//	};
		//	var loadControlAnalyzer = loadControlAnalyzerBuilder.Build();
		//	var staticAnalyzer = new StaticAnalyzer(algebraicModel, problem, loadControlAnalyzer);

		//	loadControlAnalyzer.IncrementalDisplacementsLog = new IncrementalDisplacementsLog(
		//		new List<(INode node, IDofType dof)>()
		//		{
		//			(model.NodesDictionary[20], StructuralDof.TranslationX)
		//		}, algebraicModel
		//	);

		//	staticAnalyzer.Initialize();
		//	staticAnalyzer.Solve();

		//	return loadControlAnalyzer.IncrementalDisplacementsLog;
		//}
		private static DOFSLog SolveModel(Model model)
		{
			var solverFactory = new SkylineSolver.Factory();
			var algebraicModel = solverFactory.BuildAlgebraicModel(model);
			var solver = solverFactory.BuildSolver(algebraicModel);
			var problem = new ProblemStructural(model, algebraicModel);

			var loadControlAnalyzerBuilder = new LoadControlAnalyzer.Builder(algebraicModel, solver, problem, numIncrements: 100)
			{
				ResidualTolerance = 1E-3,
				MaxIterationsPerIncrement = 100,
				NumIterationsForMatrixRebuild = 1
			};
			var loadControlAnalyzer = loadControlAnalyzerBuilder.Build();
			var staticAnalyzer = new StaticAnalyzer(algebraicModel, problem, loadControlAnalyzer);

			watchDofs.Add((model.NodesDictionary[20], StructuralDof.TranslationX));
			watchDofs.Add((model.NodesDictionary[20], StructuralDof.TranslationZ));
			loadControlAnalyzer.LogFactory = new LinearAnalyzerLogFactory(watchDofs, algebraicModel);

			staticAnalyzer.Initialize();
			staticAnalyzer.Solve();

			return (DOFSLog)loadControlAnalyzer.Logs[0];
		}
	}

}
