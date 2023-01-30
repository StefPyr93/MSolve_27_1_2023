using MGroup.MSolve.AnalysisWorkflow.Transient;
using MGroup.MSolve.Discretization;
using MGroup.MSolve.Discretization.Entities;

namespace MGroup.Constitutive.Structural.InitialConditions
{
	public class NodalInitialDisplacement : INodalDisplacementInitialCondition
	{
		public IStructuralDofType DOF { get; }

		public INode Node { get; }

		public double Amount { get; }

		public DifferentiationOrder DifferentiationOrder => DifferentiationOrder.Zero;

		public NodalInitialDisplacement(INode node, IStructuralDofType dofType, double amount)
		{
			this.Node = node;
			this.DOF = dofType;
			this.Amount = amount;
		}

		INodalModelQuantity<IStructuralDofType> INodalModelQuantity<IStructuralDofType>.WithAmount(double amount) => new NodalInitialDisplacement(Node, DOF, amount);
		INodalInitialCondition<IStructuralDofType> INodalInitialCondition<IStructuralDofType>.WithAmount(double amount) => new NodalInitialDisplacement(Node, DOF, amount);
	}
}
