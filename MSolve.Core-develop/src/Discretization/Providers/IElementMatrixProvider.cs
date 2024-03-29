using MGroup.LinearAlgebra.Matrices;

//TODO: Since this interface defines only a single method and the implementations are trivial, I think this interface should be 
//      integrated into a provider interface.
namespace MGroup.MSolve.Discretization.Providers
{
    public interface IElementMatrixProvider
    {
        IMatrix Matrix(IElementType element);

	}
}
