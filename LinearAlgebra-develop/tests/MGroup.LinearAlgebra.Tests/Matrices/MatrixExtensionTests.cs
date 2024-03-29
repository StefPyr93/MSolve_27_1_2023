using System.Collections.Generic;
using System.Linq;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Tests.Utilities;
using Xunit;

//TODO: either use the matrices in TestData or move the example matrices there.
namespace MGroup.LinearAlgebra.Tests.Matrices
{
	/// <summary>
	/// Tests for extension methods in <see cref="MatrixExtensions"/>.
	/// Authors: Serafeim Bakalakos
	/// </summary>
	public static class MatrixExtensionTests
	{
		private static readonly MatrixComparer comparer = new MatrixComparer(1E-13);

		private static IReadOnlyList<(double[,] matrix, double[,] rref, int[] independentCols)> CreateRrefTestData
		{
			get
			{
				var result = new List<(double[,] matrix, double[,] rref, int[] independentCols)>();

				// Example 1
				var matrix1 = new double[,]
				{
					{ 3, 1 },
					{ 3, 4 }
				};
				var rref1 = new double[,]
				{
					{ 1, 0 },
					{ 0, 1 }
				};
				var independentCols1 = new int[] { 0, 1 };
				result.Add((matrix1, rref1, independentCols1));

				// Example 2
				var matrix2 = new double[,]
				{
					{  1,  0,  1,  3 },
					{  2,  3,  4,  7 },
					{ -1, -3, -3, -4 }
				};
				var rref2 = new double[,]
				{
					{ 1, 0,     1,     3 },
					{ 0, 1, 2.0/3, 1.0/3 },
					{ 0, 0,     0,     0 }
				};
				var independentCols2 = new int[] { 0, 1 };
				result.Add((matrix2, rref2, independentCols2));

				// Example 3
				var matrix3 = new double[,]
				{
					{ 1, 2, 3, 0, 0, 0 },
					{ 0, 0, 1, 1, 0, 1 },
					{ 0, 0, 0, 1, 1, 1 }
				};
				var rref3 = new double[,]
				{
					{ 1, 2, 0, 0,  3, 0 },
					{ 0, 0, 1, 0, -1, 0 },
					{ 0, 0, 0, 1,  1, 1 }
				};
				var independentCols3 = new int[] { 0, 2, 3 };
				result.Add((matrix3, rref3, independentCols3));

				// Example 4
				var matrix4 = new double[,]
				{
					{ 8, 1, 6 },
					{ 3, 5, 7 },
					{ 4, 9, 2 }
				};
				var rref4 = new double[,]
				{
					{ 1, 0, 0 },
					{ 0, 1, 0 },
					{ 0, 0, 1 }
				};
				var independentCols4 = new int[] { 0, 1, 2 };
				result.Add((matrix4, rref4, independentCols4));

				// Example 5
				var matrix5 = new double[,]
				{
					{ 16,  2,  3, 13 },
					{  5, 11, 10,  8 },
					{  9,  7,  6, 12 },
					{  4, 14, 15,  1 }
				};
				var rref5 = new double[,]
				{
					{ 1, 0, 0,  1 },
					{ 0, 1, 0,  3 },
					{ 0, 0, 1, -3 },
					{ 0, 0, 0,  0 }
				};
				var independentCols5 = new int[] { 0, 1, 2 };
				result.Add((matrix5, rref5, independentCols5));

				return result;
			}
		}

		[Fact]
		private static void TestReducedRowEchelonForm()
		{
			foreach ((double[,] matrix, double[,] rrefExpected, int[] independentColsExpected) in CreateRrefTestData)
			{
				(Matrix rrefComputed, List<int> independentColsComputed) = Matrix.CreateFromArray(matrix).ReducedRowEchelonForm();
				comparer.AssertEqual(independentColsExpected, independentColsComputed.ToArray());
				comparer.AssertEqual(Matrix.CreateFromArray(rrefExpected), rrefComputed);
			}
		}

		[Fact]
		private static void TestSpdiags()
		{
			// See example in https://www.mathworks.com/help/matlab/ref/spdiags.html
			int[] d = { -4, -2, -1, 0, 3, 4, 5 };
			var B = Matrix.CreateFromArray(new double[6, 7]
			{
				{ 1, 1, 1, 1, 1, 1, 1 },
				{ 2, 2, 2, 2, 2, 2, 2 },
				{ 3, 3, 3, 3, 3, 3, 3 },
				{ 4, 4, 4, 4, 4, 4, 4 },
				{ 5, 5, 5, 5, 5, 5, 5 },
				{ 6, 6, 6, 6, 6, 6, 6 }
			});

			var expectedS1 = Matrix.CreateFromArray(new double[6, 6]
			{
				{ 1, 0, 0, 4, 5, 6 },
				{ 1, 2, 0, 0, 5, 6 },
				{ 1, 2, 3, 0, 0, 6 },
				{ 0, 2, 3, 4, 0, 0 },
				{ 1, 0, 3, 4, 5, 0 },
				{ 0, 2, 0, 4, 5, 6 }
			});
			Matrix computedS1 = B.Spdiags(d, 6, 6);
			comparer.AssertEqual(expectedS1, computedS1);

			var expectedS2 = Matrix.CreateFromArray(new double[5,6]
			{
				{ 1, 0, 0, 1, 1, 1 },
				{ 2, 2, 0, 0, 2, 2 },
				{ 3, 3, 3, 0, 0, 3 },
				{ 0, 4, 4, 4, 0, 0 },
				{ 5, 0, 5, 5, 5, 0 }
			});
			Matrix computedS2 = B.Spdiags(d, 5, 6);
			comparer.AssertEqual(expectedS2, computedS2);
		}
	}
}
