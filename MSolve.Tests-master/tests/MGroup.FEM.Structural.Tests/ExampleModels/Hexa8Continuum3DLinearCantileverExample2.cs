using System.Collections.Generic;
using MGroup.Constitutive.Structural;
using MGroup.Constitutive.Structural.BoundaryConditions;
using MGroup.Constitutive.Structural.Continuum;
using MGroup.Constitutive.Structural.Transient;
using MGroup.FEM.Structural.Continuum;
using MGroup.MSolve.Discretization;
using MGroup.MSolve.Discretization.Entities;


namespace MGroup.FEM.Structural.Tests.ExampleModels
{
	public class Hexa8Continuum3DLinearCantileverExample
	{
        public static readonly double expected_solution_node20_TranslationX = 0.4938;
        public static readonly double expected_solution_node20_TranslationZ = -0.086364;
        public static Model CreateModel()
		{
			var nodeData = new double[,] {
				{-0.250000,-0.250000,-1.000000},
				{0.250000,-0.250000,-1.000000},
				{-0.250000,0.250000,-1.000000},
				{0.250000,0.250000,-1.000000},
				{-0.250000,-0.250000,-0.500000},
				{0.250000,-0.250000,-0.500000},
				{-0.250000,0.250000,-0.500000},
				{0.250000,0.250000,-0.500000},
				{-0.250000,-0.250000,0.000000},
				{0.250000,-0.250000,0.000000},
				{-0.250000,0.250000,0.000000},
				{0.250000,0.250000,0.000000},
				{-0.250000,-0.250000,0.500000},
				{0.250000,-0.250000,0.500000},
				{-0.250000,0.250000,0.500000},
				{0.250000,0.250000,0.500000},
				{-0.250000,-0.250000,1.000000},
				{0.250000,-0.250000,1.000000},
				{-0.250000,0.250000,1.000000},
				{0.250000,0.250000,1.000000}
			};
			//double correction = 10;// +20;

			var elementData = new int[,] {
				{1,8,7,5,6,4,3,1,2},
				{2,12,11,9,10,8,7,5,6},
				{3,16,15,13,14,12,11,9,10},
				{4,20,19,17,18,16,15,13,14}
			};

			var model = new Model();

			model.SubdomainsDictionary.Add(key: 0, new Subdomain(id: 0));

			for (var i = 0; i < nodeData.GetLength(0); i++)
			{
				var nodeId = i + 1;
				model.NodesDictionary.Add(nodeId, new Node(
					id: nodeId,
					x: nodeData[i, 0],
					y: nodeData[i, 1],
					z: nodeData[i, 2]));
			}

			for (var i = 0; i < elementData.GetLength(0); i++)
			{
				var nodeSet = new Node[8];
				for (var j = 0; j < 8; j++)
				{
					var nodeID = elementData[i, j + 1];
					nodeSet[j] = (Node)model.NodesDictionary[nodeID];
				}


			//	var Continuum3dElementFactory = new ContinuumElement3DFactory(
			////new ElasticMaterial3D(youngModulus: 1000 * 1e9, poissonRatio: 0.20),
			////commonDynamicProperties: new TransientAnalysisProperties(3150d, 0, 0)
			//new ElasticMaterial3D(youngModulus: 800000000d, poissonRatio: 1.0d / 3.0d),
			//commonDynamicProperties: new TransientAnalysisProperties(1, 0, 0)
			//);

				//var element = Continuum3dElementFactory.CreateElement(CellType.Hexa8, nodeSet);
				var Continuum3dElementFactory = new ContinuumElement3DFactory(new VonMisesMaterial3D(800000d, 1.0d / 3.0d, 5000d, 100000d), new TransientAnalysisProperties(1, 0, 0));    
				var element = Continuum3dElementFactory.CreateElement(CellType.Hexa8, nodeSet);
				element.ID = i + 1;

				model.ElementsDictionary.Add(element.ID, element);
				model.SubdomainsDictionary[0].Elements.Add(element);
			}

			var constraints = new List<INodalDisplacementBoundaryCondition>();
			for (var i = 1; i < 5; i++)
			{
				constraints.Add(new NodalDisplacement(model.NodesDictionary[i], StructuralDof.TranslationX, amount: 0d));
				constraints.Add(new NodalDisplacement(model.NodesDictionary[i], StructuralDof.TranslationY, amount: 0d));
				constraints.Add(new NodalDisplacement(model.NodesDictionary[i], StructuralDof.TranslationZ, amount: 0d));
			}

			var loads = new List<INodalLoadBoundaryCondition>();
			for (var i = 17; i < 21; i++)
			{
				loads.Add(new NodalLoad(model.NodesDictionary[i], StructuralDof.TranslationX, amount: 1.9 * 96d));
			}

			model.BoundaryConditions.Add(new StructuralBoundaryConditionSet(constraints, loads));

			return model;
		}

		//public static IReadOnlyList<double[]> GetExpectedDisplacements()
		//{
		//	return new double[][]
		//	{
  //              //new[] { 0.2632, -0.047514 },//elastic material
		//		//new[] {0.00034537 },
  //  //            new[] { 0.0011821 },
  //  //            new[] { 0.0025498 },
  //  //            new[] { 0.0043864 },
  //  //            new[] { 0.0065511 },
  //  //            new[] { 0.0088655 },
  //  //            new[] { 0.011156  },
  //  //            new[] { 0.013289 },
  //  //            new[] { 0.015185 },
  //  //            new[] { 0.016831 },
  //  //            new[] { 0.018261 },
  //  //            new[] { 0.019548 },
  //  //            new[] { 0.020776 },
  //  //            new[] { 0.022027 },
  //  //            new[] { 0.02336 },
  //  //            new[] { 0.024807 },
  //  //            new[] { 0.026372 },
  //  //            new[] { 0.028036 },
  //  //            new[] { 0.029761 },
  //  //            new[] { 0.031508 },
  //  //            new[] { 0.033239 },
  //  //            new[] { 0.034929 },
  //  //            new[] { 0.036563 },
  //  //            new[] { 0.038142 },
  //  //            new[] { 0.039677 },
  //  //            new[] { 0.041185 },
  //  //            new[] { 0.042686 },
  //  //            new[] { 0.044196 },
  //  //            new[] { 0.045727 },
  //  //            new[] { 0.047286 },
  //  //            new[] { 0.048869 },
  //  //            new[] { 0.050473 },
  //  //            new[] { 0.052088 },
  //  //            new[] { 0.053705 },
  //  //            new[] { 0.055316 },
  //  //            new[] { 0.056917 },
  //  //            new[] { 0.058506 },
  //  //            new[] { 0.060082  },
  //  //            new[] { 0.061686 },
  //  //            new[] { 0.0634  },
  //  //            new[] { 0.065305 },
  //  //            new[] { 0.067492 },
  //  //            new[] { 0.069998 },
  //  //            new[] { 0.072907 },
  //  //            new[] { 0.076287 },
  //  //            new[] { 0.080161 },
  //  //            new[] { 0.084514 },
  //  //            new[] { 0.089298 },
  //  //            new[] { 0.094438 },
  //  //            new[] { 0.10003 },
  //  //            new[] { 0.10601 },
  //  //            new[] { 0.11233 },
  //  //            new[] { 0.11893 },
  //  //            new[] { 0.12578 },
  //  //            new[] { 0.13281 },
  //  //            new[] { 0.13997 },
  //  //            new[] { 0.14717 },
  //  //            new[] { 0.15452 },
  //  //            new[] { 0.16193  },
  //  //            new[] { 0.16935 },
  //  //            new[] { 0.17674 },
  //  //            new[] { 0.18411 },
  //  //            new[] { 0.19145 },
  //  //            new[] { 0.19876 },
  //  //            new[] { 0.20603 },
  //  //            new[] { 0.21326 },
  //  //            new[] { 0.22046 },
  //  //            new[] { 0.22763 },
  //  //            new[] { 0.23478 },
  //  //            new[] { 0.24195 },
  //  //            new[] { 0.24925 },
  //  //            new[] { 0.25667 },
  //  //            new[] { 0.26421 },
  //  //            new[] { 0.27188 },
  //  //            new[] { 0.27972 },
  //  //            new[] { 0.28774 },
  //  //            new[] { 0.29596 },
  //  //            new[] { 0.294322309 },
  //  //            new[] { 0.31294 },
  //  //            new[] { 0.32168 },
  //  //            new[] { 0.33056 },
  //  //            new[] { 0.33954 },
  //  //            new[] { 0.3486 },
  //  //            new[] { 0.35771 },
  //  //            new[] { 0.36682 },
  //  //            new[] { 0.37591 },
  //  //            new[] { 0.38495 },
  //  //            new[] { 0.39392 },
  //  //            new[] { 0.380529737 },
  //  //            new[] { 0.388387312 },
  //  //            new[] { 0.396246897 },
  //  //            new[] { 0.42865 },
  //  //            new[] { 0.43703 },
  //  //            new[] { 0.44528 },
  //  //            new[] { 0.45343 },
  //  //            new[] { 0.46149 },
  //  //            new[] { 0.46954 },
  //  //            new[] { 0.47761 },
  //  //            new[] { 0.4857 },
  //              new[] {  0.4938 },
  //          };
		
	}
}
