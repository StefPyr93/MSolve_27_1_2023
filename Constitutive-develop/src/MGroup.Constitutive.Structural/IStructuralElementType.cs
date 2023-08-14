using MGroup.MSolve.Discretization;
using MGroup.LinearAlgebra.Matrices;
using MGroup.Constitutive.Structural.Continuum;
using System.Collections.Generic;

namespace MGroup.Constitutive.Structural
{
	public interface IStructuralElementType : IElementType
	{
		IMatrix StiffnessMatrix();
		IMatrix MassMatrix();
		IMatrix DampingMatrix();
	}
}
